'''Gaussian prediction using encoding.
'''

import argparse
import os
from os.path import join
import glob
import numpy as np
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import networks

""" CONFIG """

ROOT = '../..'
parser = argparse.ArgumentParser(description='Naive Model for Gaussian Prediction.')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr-pred', type=float, default=1e-4,
                    help='pred model learning rate')
parser.add_argument('--lr-enc', type=float, default=1e-4,
                    help='enc model learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--data-dir', type=str, default=join(ROOT, 'datasets/gaussian'),
                    help='path to training data')
parser.add_argument('--batch-size', type=int, default=5,
                    help='batch size')
parser.add_argument('--max-files', type=int, default=-1,
                    help='max files to load. (-1 for no max)')
parser.add_argument('--test-files', type=int, default=50,
                    help='num files to test.')
parser.add_argument('--start-train-ind', type=int, default=80,
                    help='index of each group element to start backproping.')
parser.add_argument('--start-test-ind', type=int, default=80,
                    help='index of each group element to start testing.')
parser.add_argument('--enc-size', type=int, default=25,
                    help='Size of encoding.')
parser.add_argument('--weight-ind', type=int, default=-1,
                    help='If set, print weight matrix of test set at ind')
parser.add_argument('--use-lstm', action="store_true", help='Use LSTM for Enc Model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='lstm',
                    help='model to use (lstm/lincom)')
parser.add_argument('--loss-fn', type=str, default='mse',
                    help='path to training data')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.loss_fn = args.loss_fn.lower()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

""" DATA LOADERS """
class GaussianDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        super(GaussianDataset, self).__init__()
        self.files = glob.glob(args.data_dir + '/*.npz')

        if args.max_files > 0:
            self.files = self.files[:args.max_files]

        assert len(self.files) > args.test_files

        if train:
            self.files = self.files[:-args.test_files]
        else:
            self.files = self.files[-args.test_files:]


        with np.load(self.files[0]) as data:
            self.x_size = int(data['x_size'])
            self.y_size = int(data['y_size'])
            self.data = [GaussianDataset.to_list(data)]
            if not train:
                self.weights = [data['W']]

        for file_ in self.files[1:]:
            with np.load(file_) as data:
                assert data['x_size'] == self.x_size and data['y_size'] == self.y_size
                self.data.append(GaussianDataset.to_list(data))
                if not train:
                    self.weights.append(data['W'])

        if not train:
            self.weights = np.array(self.weights)
            num_batches = int(math.ceil(len(self.weights) / float(args.batch_size)))
            self.weights = np.array_split(self.weights, num_batches, axis=0)
            self.weights = [torch.from_numpy(x.astype(np.float32)).cuda() for x in self.weights]

    @staticmethod
    def to_list(data):
        to_torch_tensor = lambda x: torch.from_numpy(x.astype(np.float32))
        return [(to_torch_tensor(x), to_torch_tensor(y)) for x, y in zip(data['x'], data['y'])]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

""" INITIALIZATIONS """

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_set = GaussianDataset(train=True)
train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_set = GaussianDataset(train=False)
test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=args.batch_size, shuffle=False, **kwargs)
x_size, y_size = train_set.x_size, train_set.y_size

pred_model = networks.PredictNet(args.enc_size, x_size, y_size)
enc_model = networks.EncNet(args.enc_size, x_size, y_size, args.use_lstm)
if args.cuda:
    pred_model.cuda()
    enc_model.cuda()

pred_optim = optim.Adam(pred_model.parameters(), lr=args.lr_pred)
enc_optim = optim.Adam(enc_model.parameters(), lr=args.lr_enc)

mse = nn.MSELoss()
l1 = nn.L1Loss()

""" HELPERS """

def zero_variable_(size, volatile=False):
    if args.cuda:
        return Variable(torch.cuda.FloatTensor(*size).zero_(), volatile=volatile)
    else:
        return Variable(torch.FloatTensor(*size).zero_(), volatile=volatile)

""" TRAIN/TEST LOOPS """

def train_epoch(epoch):
    tot_loss = 0
    num_samples = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(train_loader):
        batch_loss = zero_variable_((1,))
        pred_optim.zero_grad()
        enc_optim.zero_grad()

        batch_size = batch[0][0].size()[0] # get dynamic batch size
        enc = zero_variable_((batch_size, args.enc_size))
        encs = []

        for i, (x, y) in enumerate(batch):
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            if i >= args.start_train_ind:
                pred, W_est = pred_model(x, encs[i-1])
                if args.loss_fn == 'l1':
                    loss = l1(pred, y)
                else:
                    loss = mse(pred, y)

                batch_loss += loss
                tot_loss += loss.data[0]
                num_samples += 1

            enc = enc_model(x, y, enc)
            encs.append(enc)

        batch_loss.backward()
        pred_optim.step()
        enc_optim.step()

    print 'Time: {:.2f}s Epoch: {} Train Loss ({}): {:.3f}'.format(
        time.time() - start_time, epoch, args.loss_fn.upper(), tot_loss / num_samples)

def test_epoch(epoch):
    tot_loss = 0
    W_loss = 0
    num_samples = 0
    for batch_idx, batch in enumerate(test_loader):
        batch_size = batch[0][0].size()[0] # get actual batch size
        enc = zero_variable_((batch_size, args.enc_size))
        encs = []

        for i, (x, y) in enumerate(batch):
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x, volatile=True), Variable(y)

            if i >= args.start_test_ind:
                pred, W = pred_model(x, encs[i-1])

                if i == args.start_test_ind:
                    W_loss += l1(Variable(test_set.weights[batch_idx]), W).data[0]

                loss = l1(pred, y)
                tot_loss += loss.data[0]
                num_samples += 1

            enc = enc_model(x, y, enc)
            encs.append(enc)

    print 'Test Loss (L1): {:.3f} W Loss (L1): {:.3f}'.format(
            tot_loss / num_samples, W_loss / (batch_idx+1))

if __name__ == '__main__':
    for epoch in range(1, args.epochs+1):
        train_epoch(epoch)
        test_epoch(epoch)


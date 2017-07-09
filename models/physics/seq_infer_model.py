'''Physics Mass Inference model.
'''

print "Importing"

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

""" CONFIG """

DATAROOT = '/data/vision/oliva/scenedataset/urops/scenelayout/.physics/'
ROOT = '../..'
parser = argparse.ArgumentParser(description='Physics mass inference model.')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr-pred', type=float, default=1e-4,
                    help='pred model learning rate')
parser.add_argument('--lr-enc', type=float, default=1e-4,
                    help='enc model learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--data-dir', type=str, default=join(DATAROOT, ''),
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
parser.add_argument('--init-enc', type=float, default=7,
                    help='Scalar value to initialize encoding too')
parser.add_argument('--weight-ind', type=int, default=-1,
                    help='If set, print weight matrix of test set at ind')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Num workers to load data')
parser.add_argument('--use-lstm', action="store_true", help='Use LSTM for Enc Model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--loss-fn', type=str, default='mse',
                    help='path to training data')
parser.add_argument('--widths', type=int, default=[50, 50],
                    nargs='+', help='Size of encodings.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.loss_fn = args.loss_fn.lower()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

""" DATA LOADERS """

def collate(batch):
    x, y = zip(*batch)
    x = zip(*x)

    # allocate storage off the tensors in the first batch
    numels = [sum([tensor.numel() for tensor in tensors]) for tensors in x]
    storages = [tensors[0].storage()._new_shared(numel) for numel, tensors in zip(numels, x)]
    outs = [tensors[0].new(storage) for storage, tensors in zip(storages, x)]
    x = [torch.stack(tensors, 0, out=out) for out, tensors in zip(outs, x)]

    y = torch.Tensor(y)

    return x, y

class PhysicsDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        super(PhysicsDataset, self).__init__()
        self.files = glob.glob(args.data_dir + '/*.npz')

        if args.max_files > 0:
            self.files = self.files[:args.max_files]

        assert len(self.files) > args.test_files

        if train:
            self.files = self.files[:-args.test_files]
        else:
            self.files = self.files[-args.test_files:]


        with np.load(self.files[0]) as data:
            self.x_size = data['x'].shape[1] * data['x'].shape[2]

    @staticmethod
    def to_list(x):
        to_torch_tensor = lambda t: torch.from_numpy(t.reshape([-1]).astype(np.float32))
        return [to_torch_tensor(a) for a in x]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        with np.load(self.files[key]) as data:
            x = PhysicsDataset.to_list(data['x'])
            y = float(data['y'])

        return x, y


""" MODELS """

class SeqEncNet(nn.Module):
    def __init__(self, widths, x_size):
        super(SeqEncNet, self).__init__()

        if args.use_lstm:
            lstm_weights = [x_size] + widths[:-1]
            self.enc_lstms = nn.ModuleList([
                nn.LSTMCell(inp, out) for inp, out in zip(lstm_weights[:-1], lstm_weights[1:])])

            lin_weights = widths[-2:]
            self.enc_linears = nn.ModuleList([
                nn.Linear(inp, out) for inp, out in zip(lin_weights[:-1], lin_weights[1:])])

            self.reset_cell_state()
        else:
            lin_weights = [x_size + widths[-1]] + widths

            self.enc_linears = nn.ModuleList([
                nn.Linear(inp, out) for inp, out in zip(lin_weights[:-1], lin_weights[1:])])

    def forward(self, x, h):
        if args.use_lstm:
            h = x
        else:
            h = torch.cat((x, h), dim=1)

        if args.use_lstm:
            for i, lstm in enumerate(self.enc_lstms):
                h, c = lstm(h, self.cell_states[i])
                self.cell_states[i] = (h, c)
        for i, lin in enumerate(self.enc_linears):
            if i == len(self.enc_linears) - 1:
                h = lin(h)
            else:
                h = F.relu(lin(h))

        return h

    def reset_cell_state(self, batch_size):
        assert args.use_lstm

        self.cell_states = [(zero_variable_((batch_size, width)),
            zero_variable_((batch_size, width))) for width in args.widths[:-1]]


""" INITIALIZATIONS """

print "Initializing"

kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {'num_workers': 4}

start_time = time.time()
train_set = PhysicsDataset(train=True)
train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=args.batch_size, shuffle=True, collate_fn=collate, **kwargs)
test_set = PhysicsDataset(train=False)
test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=args.batch_size, shuffle=False, collate_fn=collate,  **kwargs)

x_size = train_set.x_size

enc_model = SeqEncNet(args.widths, x_size)
if args.cuda:
    enc_model.cuda()

enc_optim = optim.Adam(enc_model.parameters(), lr=args.lr_enc)

mse = nn.MSELoss()
l1 = nn.L1Loss()

""" HELPERS """

def zero_variable_(size, volatile=False):
    if args.cuda:
        return Variable(torch.cuda.FloatTensor(*size).zero_(), volatile=volatile)
    else:
        return Variable(torch.FloatTensor(*size).zero_(), volatile=volatile)

def init_variable_(batch_size):
    return zero_variable_((batch_size, args.widths[-1]))
    # return torch.cat([torch.add(zero_variable_((batch_size, 1)), args.init_enc), zero_variable_((batch_size, args.widths[-1] - 1))], dim=1)

""" TRAIN/TEST LOOPS """

def train_epoch(epoch):
    tot_loss = 0
    num_samples = 0
    start_time = time.time()
    for batch_idx, (x, y) in enumerate(train_loader):
        enc_optim.zero_grad()

        batch_size = x[0].size()[0]
        enc = init_variable_(batch_size)
        if args.use_lstm:
            enc_model.reset_cell_state(batch_size)

        if args.cuda:
            y = y.cuda()
        y = Variable(y)

        for i, sample in enumerate(x):
            if args.cuda:
                sample = sample.cuda()
            sample = Variable(sample)

            enc = enc_model(sample, enc)

        loss = mse(enc[:,:1], y)
        tot_loss += loss.data[0]
        num_samples += 1

        loss.backward()
        enc_optim.step()

    print 'Time: {:.2f}s Epoch: {} Train Loss ({}): {:.3f}'.format(
        time.time() - start_time, epoch, args.loss_fn.upper(), tot_loss / num_samples)

def test_epoch(epoch):
    l1_loss = 0
    mse_loss = 0
    num_samples = 0
    start_time = time.time()
    for batch_idx, (x, y) in enumerate(test_loader):
        batch_size = x[0].size()[0]
        enc = init_variable_(batch_size)
        if args.use_lstm:
            enc_model.reset_cell_state(batch_size)

        if args.cuda:
            y = y.cuda()
        y = Variable(y)

        for i, sample in enumerate(x):
            if args.cuda:
                sample = sample.cuda()
            sample = Variable(sample)

            enc = enc_model(sample, enc)

        l1_loss += l1(enc[:,:1], y).data[0]
        mse_loss += mse(enc[:,:1], y).data[0]
        num_samples += 1

    print 'Test Loss (L1): {:.3f} (MSE): {:.3f}'.format(l1_loss / num_samples, mse_loss / num_samples)

if __name__ == '__main__':
    print "Start Training"
    for epoch in range(1, args.epochs+1):
        train_epoch(epoch)
        test_epoch(epoch)


'''Physics Mass Inference model.
'''

print "Importing"

import sys
sys.path.append('..')

import argparse
import os
from os.path import join
import glob
import numpy as np
import time
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import shared.networks as networks

""" CONFIG """

DATAROOT = '/data/vision/oliva/scenedataset/urops/scenelayout/.physics_n3/'
ROOT = '../..'
parser = argparse.ArgumentParser(description='Physics mass inference model.')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr-trans', type=float, default=1e-3,
                    help='transform model learning rate')
parser.add_argument('--lr-enc', type=float, default=1e-3,
                    help='enc model learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--data-dir', type=str, default=join(DATAROOT, ''),
                    help='path to training data')
parser.add_argument('--log', type=str, default='log.txt',
                    help='Store logs.')
parser.add_argument('--batch-size', type=int, default=5,
                    help='batch size')
parser.add_argument('--max-files', type=int, default=-1,
                    help='max files to load. (-1 for no max)')
parser.add_argument('--test-files', type=int, default=50,
                    help='num files to test.')
parser.add_argument('--start-train-ind', type=int, default=20,
                    help='index of each group element to start backproping.')
parser.add_argument('--start-test-ind', type=int, default=20,
                    help='index of each group element to start testing.')
parser.add_argument('--weight-ind', type=int, default=-1,
                    help='If set, print weight matrix of test set at ind')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Num workers to load data')
parser.add_argument('--use-lstm', action="store_true", help='Use LSTM for Enc Model')
parser.add_argument('--use-prior', action="store_true", help='Use prior')
parser.add_argument('--progbar', action="store_true", help='Display progbar')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-prior', action='store_true', default=False,
                    help='disables use of prior')
parser.add_argument('--rolling', action='store_true', default=False,
                    help='trains the model on rolling updates of the encoding')
parser.add_argument('--loss-fn', type=str, default='mse',
                    help='path to training data')
parser.add_argument('--widths', type=int, default=[50, 50],
                    nargs='+', help='Size of encodings.')
parser.add_argument('--trans-widths', type=int, default=[25, 25],
                    nargs='+', help='Size of transform layer.')
parser.add_argument('--num-sequential-frames', type=int, default=4,
                    help='Number of sequential frames to use for training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.use_prior = not args.no_prior
args.loss_fn = args.loss_fn.lower()
args.enc_widths = args.widths
args.rolling = True

if args.progbar:
    progbar = tqdm
else:
    progbar = lambda x: x

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
            self.x_size = data['x'].shape[1] * data['x'].shape[2] * args.num_sequential_frames
            self.y_size = data['y'].shape[0]

    @staticmethod
    def to_list(x):
        to_torch_tensor = lambda t: torch.from_numpy(t.reshape([-1]).astype(np.float32))
        return [to_torch_tensor(a) for a in x]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        with np.load(self.files[key]) as data:
            x = []
            for i in xrange(len(data['x']) - args.num_sequential_frames):
                x.append(data['x'][i:i+args.num_sequential_frames])
            x = PhysicsDataset.to_list(x)
            y = data['y']

        return x, y

""" INITIALIZATIONS """

print "Initializing"

kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {'num_workers': 4}

start_time = time.time()
train_set = PhysicsDataset(train=True)
train_loader = torch.utils.data.DataLoader(train_set, collate_fn=collate,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_set = PhysicsDataset(train=False)
test_loader = torch.utils.data.DataLoader(test_set, collate_fn=collate,
    batch_size=args.batch_size, shuffle=False,  **kwargs)

x_size = train_set.x_size
y_size = train_set.y_size

enc_model = networks.ParallelEncNet(args.enc_widths, x_size)
enc_model_wrapper = networks.get_wrapper('confweight', enc_model, args)
trans_model = networks.TransformNet(args.enc_widths[-1], args.trans_widths,  y_size)
if args.cuda:
    enc_model.cuda()
    trans_model.cuda()

enc_optim = optim.Adam(enc_model.parameters(), lr=args.lr_enc)
trans_optim = optim.Adam(trans_model.parameters(), lr=args.lr_trans)

mse = nn.MSELoss()
l1 = nn.L1Loss()

""" HELPERS """

def log(text):
    print text
    with open(args.log, 'a') as f:
        f.write(text + '\n')

def zero_variable_(size, volatile=False):
    if args.cuda:
        return Variable(torch.cuda.FloatTensor(*size).zero_(), volatile=volatile)
    else:
        return Variable(torch.FloatTensor(*size).zero_(), volatile=volatile)

""" TRAIN/TEST LOOPS """

def train_epoch(epoch):
    tot_loss = 0
    num_batches = 0
    start_time = time.time()
    for batch_idx, (x, y) in enumerate(progbar(train_loader)):
        enc_optim.zero_grad()
        trans_optim.zero_grad()

        if args.cuda:
            y = y.cuda()
        y = Variable(y)

        encs = enc_model_wrapper(x)
        batch_loss = 0
        for enc in encs[args.start_train_ind:]:
            tf_enc = trans_model(enc)
            loss = mse(tf_enc, y)
            tot_loss += loss.data[0]
            num_batches += 1

        loss.backward()
        enc_optim.step()
        trans_optim.step()

    log('Time: {:.2f}s Epoch: {} Train Loss ({}): {:.3f}'.format(
        time.time() - start_time, epoch, args.loss_fn.upper(), tot_loss / num_batches))

def test_epoch(epoch):
    l1_loss = 0
    mse_loss = 0
    num_batches = 0
    start_time = time.time()
    for batch_idx, (x, y) in enumerate(progbar(test_loader)):
        if args.cuda:
            y = y.cuda()
        y = Variable(y)

        encs = enc_model_wrapper(x)
        for enc in encs[args.start_train_ind:]:
            tf_enc = trans_model(enc)
            loss = mse(tf_enc, y)
            l1_loss += l1(tf_enc, y).data[0]
            mse_loss += mse(tf_enc, y).data[0]
            num_batches += 1

    log('Test Loss (L1): {:.3f} (MSE): {:.3f}'.format(l1_loss / num_batches, mse_loss / num_batches))

if __name__ == '__main__':
    log("Start Training")
    for epoch in range(1, args.epochs+1):
        train_epoch(epoch)
        test_epoch(epoch)


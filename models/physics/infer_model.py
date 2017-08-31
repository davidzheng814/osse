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
import random
import time
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from loader import EncPhysicsDataset
import shared.networks as networks

""" CONFIG """

DATAROOT = '/data/vision/oliva/scenedataset/urops/scenelayout/'
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
parser.add_argument('--data-file', type=str, default=join(DATAROOT, '.physics_b_n3_t75_f120_clean.h5'),
                    help='path to training data')
parser.add_argument('--log', type=str, default='log.txt',
                    help='Store logs.')
parser.add_argument('--batch-size', type=int, default=5,
                    help='batch size')
parser.add_argument('--num-points', type=int, default=-1,
                    help='max points to use. (-1 for no max)')
parser.add_argument('--test-points', type=int, default=1000,
                    help='num files to test.')
parser.add_argument('--start-train-ind', type=int, default=50,
                    help='index of each group element to start backproping.')
parser.add_argument('--start-test-ind', type=int, default=50,
                    help='index of each group element to start testing.')
parser.add_argument('--weight-ind', type=int, default=-1,
                    help='If set, print weight matrix of test set at ind')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Num workers to load data')
parser.add_argument('--progbar', action="store_true", help='Display progbar')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-prior', action='store_true', default=False,
                    help='disables use of prior')
parser.add_argument('--loss-fn', type=str, default='mse',
                    help='path to training data')
parser.add_argument('--widths', type=int, default=[50, 50],
                    nargs='+', help='Size of encodings.')
parser.add_argument('--trans-widths', type=int, default=[25, 25],
                    nargs='+', help='Size of transform layer.')
parser.add_argument('--num-sequential-frames', type=int, default=4,
                    help='Number of sequential frames to use for training')
parser.add_argument('--model', type=str, default='parallel',
                    help='model to use for encnet (parallel/lstm)')
parser.add_argument('--depth', type=int, default=1,
                    help='default depth for certain classes of models')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.use_prior = not args.no_prior
args.loss_fn = args.loss_fn.lower()
args.enc_widths = args.widths
args.rolling = True

if args.progbar:
    progbar = lambda x: tqdm(x, leave=False)
else:
    progbar = lambda x: x

torch.manual_seed(args.seed)
random.seed(args.seed)
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

""" INITIALIZATIONS """

print "Initializing"

kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {'num_workers': 4}

start_time = time.time()
train_set = EncPhysicsDataset(args.data_file, args.num_points, args.test_points,
    args.num_sequential_frames, train=True)
train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_set = EncPhysicsDataset(args.data_file, args.num_points, args.test_points,
    args.num_sequential_frames, train=False)
test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=args.batch_size, shuffle=False, **kwargs)

x_size = train_set.x_size
y_size = train_set.y_size

if args.model == 'parallel':
    enc_model = networks.ParallelEncNet(args.enc_widths, x_size, args)
    enc_model_wrapper = networks.get_wrapper('confweight', enc_model, args)
elif args.model == 'lstm':
    assert len(args.enc_widths) == 1
    enc0_structure = [(args.enc_widths[-1], args.enc_widths[-1])] * args.depth
    enc_model = networks.LSTMEncNet(args.enc_widths[-1], x_size, depth=args.depth)
    enc_model_wrapper = networks.RecurrentWrapper(enc_model, args, enc0_structure)
else:
    raise RuntimeError("Model type {} not recognized.".format(args.model))

trans_model = networks.TransformNet(args.enc_widths[-1], args.trans_widths,  y_size)
if args.cuda:
    enc_model.cuda()
    trans_model.cuda()

enc_optim = optim.Adam(enc_model.parameters(), lr=args.lr_enc)
trans_optim = optim.Adam(trans_model.parameters(), lr=args.lr_trans)

mse = nn.MSELoss()
l1 = nn.L1Loss()

print "Enc net params:", networks.num_params(enc_model_wrapper)
print "Trans net params:", networks.num_params(trans_model)

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

# Prints tensor/variable as list
def d(t):
    if isinstance(t, torch.Tensor) or isinstance(t, torch.cuda.FloatTensor):
        return d(t.cpu().numpy())
    elif isinstance(t, Variable):
        return d(t.data)
    elif isinstance(t, np.ndarray):
        if len(t) == 1:
            return d(t[0])
        return list(t)
    else:
        raise RuntimeError("Unrecognized type {}".format(type(t)))

""" TRAIN/TEST LOOPS """

def train_epoch(epoch):
    tot_loss = 0
    num_batches = 0
    start_time = time.time()
    compute_time = 0.
    loader = train_loader
    tf_encs = []
    ys = []
    for batch_idx, (x, y) in enumerate(progbar(loader)):
        enc_optim.zero_grad()
        trans_optim.zero_grad()

        if args.cuda:
            y = y.cuda()
        y = Variable(y)

        start_compute = time.time()
        encs = enc_model_wrapper(x)
        batch_loss = 0

        # Encoding at step args.start_train_ind
        enc = encs[args.start_train_ind]
        tf_enc = trans_model(enc)
        loss = mse(tf_enc, y)
        tot_loss += loss.data[0]
        tf_encs.append(tf_enc.data.cpu())
        ys.append(y.data.cpu())
        num_batches += 1

        loss.backward()
        enc_optim.step()
        trans_optim.step()

        compute_time += time.time() - start_compute

    tf_encs = torch.cat(tf_encs, dim=0)
    ys = torch.cat(ys, dim=0)

    print("Pred mean/std/min/max:", d(torch.mean(tf_encs, 0)), d(torch.std(tf_encs, 0)),
            d(torch.min(tf_encs, 0)[0]), d(torch.max(tf_encs, 0)[0]))
    print(" Act mean/std/min/max:", d(torch.mean(ys, 0)), d(torch.std(ys, 0)),
            d(torch.min(ys, 0)[0]), d(torch.max(ys, 0)[0]))

    log('Epoch: {}, Train Loss ({}): {:.3f}, Time: {:.2f}s, Compute: {:.2f}s'.format(
        epoch, args.loss_fn.upper(), tot_loss / num_batches, time.time() - start_time, 
        compute_time))

def test_epoch(epoch):
    l1_loss = 0
    mse_loss = 0
    num_batches = 0
    start_time = time.time()
    loader = test_loader
    tf_encs = []
    ys = []
    for batch_idx, (x, y) in enumerate(progbar(loader)):
        if args.cuda:
            y = y.cuda()
        y = Variable(y)

        encs = enc_model_wrapper(x)

        # Encoding at step args.start_train_ind
        enc = encs[args.start_train_ind]
        tf_enc = trans_model(enc)
        loss = mse(tf_enc, y)
        l1_loss += l1(tf_enc, y).data[0]
        mse_loss += mse(tf_enc, y).data[0]
        tf_encs.append(tf_enc.data.cpu())
        ys.append(y.data.cpu())
        num_batches += 1

    tf_encs = torch.cat(tf_encs, dim=0)
    ys = torch.cat(ys, dim=0)

    print("Pred mean/std/min/max:", d(torch.mean(tf_encs, 0)), d(torch.std(tf_encs, 0)),
            d(torch.min(tf_encs, 0)[0]), d(torch.max(tf_encs, 0)[0]))
    print(" Act mean/std/min/max:", d(torch.mean(ys, 0)), d(torch.std(ys, 0)),
            d(torch.min(ys, 0)[0]), d(torch.max(ys, 0)[0]))

    #log('Epoch: {}, Test Loss (L1): {:.3f}, (MSE): {:.3f}'.format(epoch,
    #    l1_loss / num_batches, mse_loss / num_batches))
    log('Epoch: {},  Test Loss (MSE): {:.3f}, Test Loss (L1): {:.3f}, Time: {:.2f}s'.format(
        epoch, mse_loss / num_batches, l1_loss / num_batches, time.time() - start_time))

if __name__ == '__main__':
    log("Start Training")
    for epoch in range(1, args.epochs+1):
        train_epoch(epoch)
        test_epoch(epoch)


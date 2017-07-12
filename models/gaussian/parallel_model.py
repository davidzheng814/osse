'''Gaussian prediction using encoding with parallel.
'''
import sys
sys.path.append('..')

import argparse
import os
from os.path import join
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import shared.networks as networks
import shared.util as util
import datasets

""" CONFIG """

ROOT = '../..'
parser = argparse.ArgumentParser(description='Naive Model for Gaussian Prediction.')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr-pred', type=float, default=1e-3,
                    help='pred model learning rate')
parser.add_argument('--lr-enc', type=float, default=1e-3,
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
parser.add_argument('--weight-ind', type=int, default=-1,
                    help='If set, print weight matrix of test set at ind')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--rolling', action='store_true', default=False,
                    help='whether or not using a rolling wrapper')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='normalize inputs and outputs to [-1, 1]')
parser.add_argument('--use-prior', action='store_true', default=False,
                    help='use a prior for the parallel encnet (enc0, conf0)')
parser.add_argument('--test-on-train', action='store_true', default=False,
                    help='test model on training data')
parser.add_argument('--enc-model', type=str, default='parallel',
                    help='encoder model to use (rnn/lstm/parallel)')
parser.add_argument('--enc-wrapper', type=str, default='confweight',
                    help='wrapper to use (confweight/recurrent/rollingconf/rollingrec)')
parser.add_argument('--pred-model', type=str, default='basic',
                    help='predictive model to use (basic/nonlinear)')
parser.add_argument('--loss-fn', type=str, default='mse',
                    help='path to training data')
parser.add_argument('--enc-widths', type=int, default=[25, 25],
                    nargs='+', help='Size of encoder layers. Last width determines encoding size.')
parser.add_argument('--pred-widths', type=int, default=[25, 25],
                    nargs='+', help='Size of predictor layers (for nonlinear).')
parser.add_argument('--dropout', type=float, default=0, help='Size of encoding.')
parser.add_argument('--log', type=str, default="log.txt",
                    help='log file')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.loss_fn = args.loss_fn.lower()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

""" INITIALIZATIONS """

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_set = datasets.GaussianDataset(args, train=True, test_on_train=args.test_on_train, normalize=args.normalize)
train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_set = datasets.GaussianDataset(args, train=False, test_on_train=args.test_on_train, normalize=args.normalize)
test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=args.batch_size, shuffle=False, **kwargs)
x_size, y_size = train_set.x_size, train_set.y_size

print "Data loaded."

pred_model = networks.get_predictor(args.pred_model, args, x_size, y_size)
enc_model = networks.get_encoder(args.enc_model, args, x_size, y_size)
enc_model_wrapper = networks.get_wrapper(args.enc_wrapper, enc_model, args)
if args.cuda:
    pred_model.cuda()
    enc_model_wrapper.cuda()

pred_optim = optim.Adam(pred_model.parameters(), lr=args.lr_pred)
enc_optim = optim.Adam(enc_model_wrapper.parameters(), lr=args.lr_enc)

mse = nn.MSELoss()
l1 = nn.L1Loss()

print "Models built."

""" TRAIN/TEST LOOPS """

def train_epoch(epoch):
    tot_loss = 0
    tot_loss_l1 = 0
    num_samples = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(train_loader):
        batch_loss = util.zero_variable((1,), args.cuda)
        pred_optim.zero_grad()
        enc_optim.zero_grad()

        x_batch, y_batch = zip(*batch)
        sample_batch = torch.cat([torch.stack(x_batch), torch.stack(y_batch)], dim=2)

        if enc_model_wrapper.is_rolling():
            encs = enc_model_wrapper(sample_batch) 
        else:
            enc = enc_model_wrapper(sample_batch[:args.start_train_ind]) 

        for i, (x, y) in enumerate(zip(x_batch, y_batch)[args.start_train_ind:]):
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            if enc_model_wrapper.is_rolling():
                pred, W_est = pred_model(x, encs[i])
            else:
                pred, W_est = pred_model(x, enc)
            if args.loss_fn == 'l1':
                loss = l1(pred, y)
            else:
                loss = mse(pred, y)
            batch_loss += loss
            tot_loss += loss.data[0]
            tot_loss_l1 += l1(pred,y).data[0]
            num_samples += 1

        batch_loss.backward()
        pred_optim.step()
        enc_optim.step()

    print 'Time: {:.2f}s Epoch: {} Train Loss ({}): {:.3f} Train L1: {}'.format(
        time.time() - start_time, epoch, args.loss_fn.upper(), tot_loss / num_samples,
        tot_loss_l1 / num_samples)

def test_epoch(epoch):
    tot_loss_l1 = 0
    tot_loss_mse = 0
    tot_loss_l1_gt = 0
    W_loss = 0
    num_samples = 0
    for batch_idx, batch in enumerate(test_loader):
        x_batch, y_batch = zip(*batch)
        sample_batch = torch.cat([torch.stack(x_batch), torch.stack(y_batch)], dim=2)

        if enc_model_wrapper.is_rolling():
            encs = enc_model_wrapper(sample_batch) 
        else:
            enc = enc_model_wrapper(sample_batch[:args.start_train_ind]) 

        for i, (x, y) in enumerate(zip(x_batch, y_batch)[args.start_test_ind:]):
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x, volatile=True), Variable(y, volatile=True)
            if enc_model_wrapper.is_rolling():
                pred, W_est = pred_model(x, encs[i])
            else:
                pred, W_est = pred_model(x, enc)
            
            loss_l1 = l1(pred, y)
            tot_loss_l1 += loss_l1.data[0]
            loss_mse = mse(pred, y)
            tot_loss_mse += loss_mse.data[0]

            real_W = Variable(test_set.weights[batch_idx])
            x = x.unsqueeze(1)
            pred_gt = torch.bmm(x, real_W)
            loss_l1_gt = l1(pred_gt, y)
            tot_loss_l1_gt += loss_l1_gt.data[0]

            num_samples += 1

    print 'Test Loss (L1): {:.3f} Test Loss (mse): {:.3f}'.format(
            tot_loss_l1 / num_samples, tot_loss_mse / num_samples)
    print 'GT Loss(L1): {:.3f}'.format(tot_loss_l1_gt / num_samples)

""" RUN """

if __name__ == '__main__':
    for epoch in range(1, args.epochs+1):
        train_epoch(epoch)
        test_epoch(epoch)


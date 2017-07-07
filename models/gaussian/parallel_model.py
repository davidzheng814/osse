'''Gaussian prediction using encoding with parallel.
'''

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

import networks
import datasets

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
parser.add_argument('--weight-ind', type=int, default=-1,
                    help='If set, print weight matrix of test set at ind')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='normalize inputs and outputs to [-1, 1]')
parser.add_argument('--use-prior', action='store_true', default=False,
                    help='use a prior for the parllel encnet (enc0, conf0)')
parser.add_argument('--test-on-train', action='store_true', default=False,
                    help='test model on training data')
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
enc_model = networks.get_encoder('parallel', args, x_size, y_size)
enc_model_wrapper = networks.ConfidenceWeightWrapper(
        enc_model, use_cuda=args.cuda, use_prior=args.use_prior)
if args.cuda:
    pred_model.cuda()
    enc_model.cuda()

pred_optim = optim.Adam(pred_model.parameters(), lr=args.lr_pred)
enc_optim = optim.Adam(enc_model.parameters(), lr=args.lr_enc)

mse = nn.MSELoss()
l1 = nn.L1Loss()

print "Models built."

""" HELPERS """

def zero_variable_(size, volatile=False):
    if args.cuda:
        return Variable(torch.cuda.FloatTensor(*size).zero_(), volatile=volatile)
    else:
        return Variable(torch.FloatTensor(*size).zero_(), volatile=volatile)

""" TRAIN/TEST LOOPS """

def train_epoch_wrapper(epoch):
    tot_loss = 0
    tot_loss_l1 = 0
    num_samples = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(train_loader):
        batch_loss = zero_variable_((1,))
        pred_optim.zero_grad()
        enc_optim.zero_grad()

        x_batch, y_batch = zip(*batch)
        enc = enc_model_wrapper(x_batch[:args.start_train_ind],
                                y_batch[:args.start_train_ind]) 
        for (x, y) in zip(x_batch, y_batch)[args.start_train_ind:]:
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
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

def train_epoch(epoch):
    tot_loss = 0
    tot_loss_l1 = 0
    num_samples = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(train_loader):
        batch_loss = zero_variable_((1,))
        pred_optim.zero_grad()
        enc_optim.zero_grad()

        batch_size = batch[0][0].size()[0] # get dynamic batch size

        ''' # code to randomly permute
        x, y = zip(*[(x.numpy(), y.numpy()) for x, y in batch])
        x, y = np.array(x), np.array(y)
        x, y = np.transpose(x, [1,0,2]), np.transpose(y, [1,0,2])
        # x now in shape (batch_size, seq_len, x_size)
        perm = np.random.permutation(x.shape[1])
        x, y = x[:,perm,:], y[:,perm,:]
        raise NotImplementedError
        '''

        encs = []
        confs = []
        if args.use_prior:
            enc0_expand = enc_model.enc0.repeat(batch_size,1)
            conf0_expand = enc_model.conf0.repeat(batch_size,1)
            encs.append(enc0_expand * conf0_expand)
            confs.append(conf0_expand)
        enc = None

        for i, (x, y) in enumerate(batch):
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            if i >= args.start_train_ind:
                if enc is None:
                    enc_sum = torch.stack(encs).sum(0)[0]
                    conf_sum = torch.stack(confs).sum(0)[0]
                    enc = enc_sum / conf_sum
                pred, W_est = pred_model(x, enc)
                if args.loss_fn == 'l1':
                    loss = l1(pred, y)
                else:
                    loss = mse(pred, y)

                batch_loss += loss
                tot_loss += loss.data[0]
                tot_loss_l1 += l1(pred,y).data[0]
                num_samples += 1
            else:
                local_enc, conf = enc_model(x, y)
                encs.append(local_enc * conf)
                confs.append(conf)

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
        batch_size = batch[0][0].size()[0] # get actual batch size
        encs = []
        confs = []
        if args.use_prior:
            enc0_expand = enc_model.enc0.repeat(batch_size,1)
            conf0_expand = enc_model.conf0.repeat(batch_size,1)
            encs.append(enc0_expand)
            confs.append(conf0_expand)
        enc = None

        for i, (x, y) in enumerate(batch):
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x, volatile=True), Variable(y)

            if i >= args.start_train_ind:
                if enc is None:
                    enc_sum = torch.stack(encs).sum(0)[0]
                    conf_sum = torch.stack(confs).sum(0)[0]
                    enc = enc_sum / conf_sum
                pred, W = pred_model(x, enc)
                if i == args.start_test_ind:
                    #W_loss += l1(Variable(test_set.weights[batch_idx]), W).data[0]
                    pass

                loss_l1 = l1(pred, y)
                tot_loss_l1 += loss_l1.data[0]
                loss_mse = mse(pred, y)
                tot_loss_mse += loss_mse.data[0]

                # This is how well the actual W does
                real_W = Variable(test_set.weights[batch_idx])
                x = x.unsqueeze(1)
                pred_gt = torch.bmm(x, real_W)
                loss_l1_gt = l1(pred_gt, y)
                tot_loss_l1_gt += loss_l1_gt.data[0]

                num_samples += 1
            else:
                local_enc, conf = enc_model(x, y)
                encs.append(local_enc * conf)
                confs.append(conf)

    print 'Test Loss (L1): {:.3f} Test Loss (mse): {:.3f}'.format(
            tot_loss_l1 / num_samples, tot_loss_mse / num_samples)
    print 'GT Loss(L1): {:.3f}'.format(tot_loss_l1_gt / num_samples)

if __name__ == '__main__':
    for epoch in range(1, args.epochs+1):
        train_epoch(epoch)
        test_epoch(epoch)


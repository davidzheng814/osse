'''Gaussian prediction using encoding.
'''

import argparse
import os
from os.path import join
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import networks
import datasets
import util

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
parser.add_argument('--normalize', action='store_true', default=False,
                    help='normalize inputs and outputs to [-1, 1]')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--enc-model', type=str, default='ff',
                    help='encoder model to use (ff/lstm/parallel)')
parser.add_argument('--pred-model', type=str, default='basic',
                    help='predictive model to use (basic)')
parser.add_argument('--loss-fn', type=str, default='mse',
                    help='path to training data')
parser.add_argument('--widths', type=int, default=[25, 25],
                    nargs='+', help='Size of encoding.')
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
train_set = datasets.GaussianDataset(args, train=True, normalize=args.normalize)
train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_set = datasets.GaussianDataset(args, train=False, normalize=args.normalize)
test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=args.batch_size, shuffle=False, **kwargs)
x_size, y_size = train_set.x_size, train_set.y_size

pred_model = networks.get_predictor(args.pred_model, args, x_size, y_size)
enc_model = networks.get_encoder(args.enc_model, args, x_size, y_size)

if args.cuda:
    pred_model.cuda()
    enc_model.cuda()

pred_optim = optim.Adam(pred_model.parameters(), lr=args.lr_pred)
enc_optim = optim.Adam(enc_model.parameters(), lr=args.lr_enc)

mse = nn.MSELoss()
l1 = nn.L1Loss()

""" HELPERS """

def log(text):
    print text
    with open(args.log, 'a') as f:
        f.write(text + "\n")

""" TRAIN/TEST LOOPS """

def train_epoch(epoch):
    enc_model.train()
    tot_loss = 0
    num_samples = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(train_loader):
        batch_loss = util.zero_variable((1,))
        pred_optim.zero_grad()
        enc_optim.zero_grad()

        batch_size = batch[0][0].size()[0] # get dynamic batch size
        enc = util.zero_variable((batch_size, args.enc_widths[-1]), args.cuda)
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

    log('Time: {:.2f}s Epoch: {} Train Loss ({}): {:.3f}'.format(
        time.time() - start_time, epoch, args.loss_fn.upper(), tot_loss / num_samples))

def test_epoch(epoch):
    enc_model.eval()
    tot_loss = 0
    W_loss = 0
    num_samples = 0
    for batch_idx, batch in enumerate(test_loader):
        batch_size = batch[0][0].size()[0] # get actual batch size
        enc = util.zero_variable((batch_size, args.enc_widths[-1]), args.cuda)
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

    log('Test Loss (L1): {:.3f} W Loss (L1): {:.3f}'.format(
            tot_loss / num_samples, W_loss / (batch_idx+1)))

if __name__ == '__main__':
    for epoch in range(1, args.epochs+1):
        train_epoch(epoch)
        test_epoch(epoch)


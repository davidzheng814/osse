from os.path import join
import os
import argparse
import glob
import time
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# TODO Use rollouts with discounting.
# TODO Add noise.

""" CONFIG """

DATAROOT = '/data/vision/oliva/scenedataset/urops/scenelayout/'
ROOT = '../..'
parser = argparse.ArgumentParser(description='Physics mass inference model.')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr-pred', type=float, default=5e-4,
                    help='pred model learning rate')
parser.add_argument('--alpha', type=float, default=6e5,
                    help='pred model learning rate alpha decay')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs')
parser.add_argument('--data-dir', type=str, default=join(DATAROOT, '.physics_n3'),
                    help='path to training data')
parser.add_argument('--log', type=str, default='log.txt',
                    help='Store logs.')
parser.add_argument('--batch-size', type=int, default=5,
                    help='batch size')
parser.add_argument('--max-files', type=int, default=-1,
                    help='max files to load. (-1 for no max)')
parser.add_argument('--test-files', type=int, default=50,
                    help='num files to test.')
parser.add_argument('--num-workers', type=int, default=16,
                    help='Num workers to load data')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--loss-fn', type=str, default='mse',
                    help='path to training data')
parser.add_argument('--code-size', type=int, default=64,
                    help='Size of code.')
parser.add_argument('--offsets', type=int, default=[1, 2, 4],
                    nargs='+', help='The timestep offset values.')
parser.add_argument('--n-frames', type=int, default=2,
                    help='Number of frames to combine before prediction.')
parser.add_argument('--rollout-ind', type=int,
                    help="First predicted sample used as input in prediction of the next.")
parser.add_argument('--predict-ind', type=int, default=0,
                    help="First sample used as prediction.")
parser.add_argument('--max-samples', type=int,
                    help='max samples per group.')
parser.add_argument('--anneal', action='store_true', help='Set learning rate annealing.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.loss_fn = args.loss_fn.lower()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
num_devices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

""" DATA LOADERS """

class PhysicsDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        super(PhysicsDataset, self).__init__()
        self.files = glob.glob(args.data_dir + '/*.npz')

        if args.max_files > 0:
            self.files = self.files[:args.max_files]

        assert len(self.files) > args.test_files

        self.train = train
        if train:
            self.files = self.files[:-args.test_files]
        else:
            self.files = self.files[-args.test_files:]

        with np.load(self.files[0]) as data:
            self.n_objects = data['x'].shape[1] 
            self.state_size = data['x'].shape[2] + 1 # Add 1 for mass

    @staticmethod
    def to_list(x, masses):
        masses = np.tile(masses, (x.shape[0], 1))
        masses = np.reshape(masses, (x.shape[0], -1, 1))
        x = np.concatenate([x, masses], axis=2)
        x = list(x.astype(np.float32))
        return x

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        with np.load(self.files[key]) as data:
            x = PhysicsDataset.to_list(data['x'], data['y'])

        if args.max_samples:
            if self.train:
                rand_ind = random.randint(0, len(x)-args.max_samples)
                x = x[rand_ind:rand_ind+args.max_samples]
            else:
                x = x[:args.max_samples]

        return x


class MLP(nn.Module):
    def __init__(self, widths, reshape=False, relu=True, n_objects=None):
        """ Only set n_objects if reshape = True"""
        super(MLP, self).__init__()

        self.reshape = reshape
        self.relu = relu
        self.n_objects = n_objects
        self.widths = widths
        self.linears = nn.ModuleList([
            nn.Linear(inp, out) for inp, out in zip(widths[:-1], widths[1:])])

    def forward(self, x):
        if self.reshape:
            h = x.view(-1, self.widths[0])
        else:
            h = x

        for i, lin in enumerate(self.linears):
            h = lin(h)
            if self.relu and i != len(self.linears) - 1:
                h = F.relu(h)

        if self.reshape:
            h = h.view(-1, self.n_objects, self.widths[-1])

        return h

class RelationNet(nn.Module):
    def __init__(self, n_objects, code_size):
        """Cross product concatenation. 
        Takes in code [batch_size, n_objects, code_size].
        Returns code [batch_size, n_objects, code_size].
        """
        super(RelationNet, self).__init__()

        self.n_objects = n_objects
        self.code_size = code_size
        self.inp_size = 2 * code_size
        self.MLP = MLP([self.inp_size, code_size, code_size, code_size])

        index = []
        inds = range(n_objects)
        for i in range(1, n_objects):
            inds = inds[-1:] + inds[:-1]
            index.extend(inds)

        if args.cuda:
            self.index = Variable(torch.cuda.LongTensor(index))
        else:
            self.index = Variable(torch.LongTensor(index))

    def forward(self, x):
        base = x.repeat(1, self.n_objects-1, 1)
        scrambled = torch.index_select(base, 1, self.index)

        h = torch.cat([base, scrambled], dim=2)
        h = h.view(-1, self.inp_size)
        h = self.MLP(h)
        h = h.view(-1, self.n_objects-1, self.n_objects, self.code_size)
        h = torch.sum(h, 1).view(-1, self.code_size)

        return h

class InteractionNet(nn.Module):
    def __init__(self, n_objects, code_size):
        super(InteractionNet, self).__init__()

        self.n_objects = n_objects
        self.code_size = code_size

        self.re_net = RelationNet(n_objects, code_size)
        self.sd_net = MLP([code_size, code_size, code_size])
        self.aff_net = MLP([code_size, code_size, code_size, code_size])
        self.agg_net = MLP([2*code_size, 32, code_size])

    def forward(self, x):
        pair_enc = self.re_net(x.view(-1, self.n_objects, self.code_size))
        ind_enc = self.sd_net(x)
        g_enc = self.aff_net(ind_enc + pair_enc)

        h = torch.cat([x, g_enc], dim=1)
        h = self.agg_net(h)

        return h

class StateCodeModel(nn.Module):
    def __init__(self, n_frames, state_size, code_size, n_objects):
        super(StateCodeModel, self).__init__()

        self.state_size = state_size
        self.code_size = code_size
        self.n_frames = n_frames
        self.linear = nn.Linear(state_size, code_size)
        self.mlp = MLP([n_frames*code_size, code_size, code_size])

    def forward(self, x):
        x = [self.linear(inp) for inp in x]
        h = torch.cat(x, dim=1)
        h = self.mlp(h)

        return h

class PredictNet(nn.Module):
    def __init__(self, n_objects, code_size, num_offsets):
        super(PredictNet, self).__init__()

        self.n_objects = n_objects
        self.code_size = code_size
        self.num_offsets = num_offsets

        self.inets = nn.ModuleList([
            InteractionNet(n_objects, code_size) for _ in range(num_offsets)])

        self.agg = MLP([num_offsets * code_size, code_size, code_size])

    def forward(self, inps):
        preds = []
        for inp, inet in zip(inps, self.inets):
            preds.append(inet(inp))

        h = torch.cat(preds, dim=1)
        h = self.agg(h)

        return h

kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {'num_workers': 4}

train_set = PhysicsDataset(train=True)
train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_set = PhysicsDataset(train=False)
test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=args.batch_size, shuffle=False, **kwargs)

num_offsets = len(args.offsets)
n_objects = train_set.n_objects
state_size = train_set.state_size
pred_model = PredictNet(n_objects, args.code_size, num_offsets)
if num_devices > 1:
    pred_model = nn.DataParallel(pred_model, device_ids=range(num_devices))
state_to_code_model = StateCodeModel(args.n_frames, state_size, args.code_size, n_objects)
code_to_state_model = nn.Linear(args.code_size, state_size)

if args.cuda:
    pred_model.cuda()
    state_to_code_model.cuda()
    code_to_state_model.cuda()

def set_lr(lr):
    global pred_optim
    pred_optim = optim.Adam(
            list(pred_model.parameters()) +
            list(state_to_code_model.parameters()) +
            list(code_to_state_model.parameters()),
            lr=lr)

lr_lambda = lambda epoch: args.lr_pred * max(0.03, math.e ** (-(epoch-1) * len(train_set) / args.alpha))

mse = nn.MSELoss()
l1 = nn.L1Loss()

""" HELPERS """

def zero_variable_(size, volatile=False):
    if args.cuda:
        return Variable(torch.cuda.FloatTensor(*size).zero_(), volatile=volatile)
    else:
        return Variable(torch.FloatTensor(*size).zero_(), volatile=volatile)

def log(text):
    text = str(text)
    print text
    with open(args.log, 'a') as f:
        f.write(text + '\n')

""" TRAIN/TEST LOOPS """
def process_batch(x, train):
    if train:
        pred_optim.zero_grad()

    if args.cuda:
        x = [samp.cuda() for samp in x]
    x = [Variable(samp, volatile=not train) for samp in x]
    x = [samp.view(-1, state_size) for samp in x]

    mse_loss = zero_variable_((1,), volatile=not train)
    if not train:
        l1_loss = zero_variable_((1,), volatile=not train)
        base_l1_loss = zero_variable_((1,), volatile=not train)

    aux_loss = 0
    # Add auxiliary losses
    codes = [None for _ in range(args.n_frames-1)]
    for start_ind in range(len(x)-args.n_frames+1):
        inp = x[start_ind:start_ind+args.n_frames]
        out = inp[-1]
        code = state_to_code_model(inp)
        state = code_to_state_model(code)

        # aux_loss_ = mse(state, out)
        # mse_loss += aux_loss_
        # aux_loss += aux_loss_.data[0]

        codes.append(code)

    # Add prediction losses
    num_preds = 0
    for i in range(max(args.offsets)+args.n_frames-1, len(codes)):
        # i equals current prediction ind
        true_state = x[i]
        inps = [codes[i-offset] for offset in args.offsets]
        pred_code = pred_model(inps)
        pred = code_to_state_model(pred_code)

        if args.rollout_ind and i >= args.rollout_ind:
            codes[i] = pred_code

        mse_loss += mse(pred+x[i-1], true_state)
        if not train:
            l1_loss += l1(pred+x[i-1], true_state)
            base_l1_loss += l1(x[i-1], true_state)
        num_preds += 1

    if train:
        mse_loss.backward()
        pred_optim.step()

    if train:
        return mse_loss.data[0], aux_loss
    else:
        return mse_loss.data[0], aux_loss, l1_loss.data[0], base_l1_loss.data[0], num_preds

def train_epoch(epoch):
    lr = lr_lambda(epoch) if args.anneal else args.lr_pred
    log('New learning rate: {:.6f}'.format(lr))
    set_lr(lr)

    mse_loss, aux_loss, num_batches = 0, 0, 0
    start_time = time.time()
    for batch_idx, x in enumerate(train_loader):
        mse_loss_, aux_loss_ = process_batch(x, train=True)
        mse_loss += mse_loss_
        aux_loss += aux_loss_
        num_batches += 1

    log('Time: {:.2f}s Epoch: {} Train Loss ({}): {:.3f} Aux {:.3f}'.format(
        time.time() - start_time, epoch, args.loss_fn.upper(), mse_loss / num_batches, aux_loss / num_batches))

def test_epoch(epoch):
    mse_loss, aux_loss, l1_loss, base_l1_loss, num_batches, num_preds = 0, 0, 0, 0, 0, 0
    start_time = time.time()
    for batch_idx, x in enumerate(test_loader):
        mse_loss_, aux_loss_, l1_loss_, base_l1_loss_, num_preds_ = process_batch(x, train=False)
        num_preds += num_preds_
        mse_loss += mse_loss_
        aux_loss += aux_loss_
        l1_loss += l1_loss_
        base_l1_loss += base_l1_loss_
        num_batches += 1

    log('Test Loss (L1): {:.3f} Base (L1): {:.3f} (MSE): {:.3f} Aux {:.3f}'.format(
        l1_loss / num_preds, base_l1_loss / num_preds, mse_loss / num_batches, aux_loss / num_batches))

if __name__ == '__main__':
    log("Start Training")
    for epoch in range(1, args.epochs+1):
        train_epoch(epoch)
        test_epoch(epoch)


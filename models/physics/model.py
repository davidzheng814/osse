from __future__ import print_function

import sys
sys.path.append('..')

from os.path import join
import os
import argparse
import glob
import time
import math
import random
import json
import datetime

import git
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from loader import PhysicsDataset, get_loader_or_cache
import shared.networks as networks

# TODO Use more correct discounting (more in line with paper's discount factors and increments?).
# TODO Add supervision signal on top of rollout. 
# TODO Add noise.
# TODO Perhaps train on large time sample horizons?
# TODO play with frame rate as well. 
# TODO Debug why collisions are always shown.
# TODO Maybe increase model capacity?
# TODO Maybe do a bug test.

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
parser.add_argument('--data-file', type=str, default=join(DATAROOT, '.physics_b_n3_t75_f120_clean.h5'),
                    help='path to training data')
parser.add_argument('--log', action="store_true",
                    help='Store logs.')
parser.add_argument('--log-dir', type=str, default='logs/',
                    help='Log directory.')
parser.add_argument('--batch-size', type=int, default=2000,
                    help='batch size')
parser.add_argument('--num-points', type=int, default=-1,
                    help='max points to use. (-1 for no max)')
parser.add_argument('--test-points', type=int, default=1000,
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
parser.add_argument('--num-encode', type=int, default=50,
                    help='Number of timesteps to use to encode.')
parser.add_argument('--n-enc-frames', type=int, default=4,
                    help='Number of frames to combine during prediction at each time step.')
parser.add_argument('--enc-widths', type=int, default=[20, 20, 6],
                    nargs='+', help='The timestep offset values.')
parser.add_argument('--n-frames', type=int, default=2,
                    help='Number of frames to combine during prediction at each time step.')
parser.add_argument('--discount', action="store_true",
                    help="Whether to discount future rollout states at first.")
parser.add_argument('---beta', type=float, default=1.5e5,
                    help="The rollout discount exponent. See code.")
parser.add_argument('--predict-ind', type=int, default=0,
                    help="First sample used as prediction.")
parser.add_argument('--max-samples', type=int,
                    help='max samples per group.')
parser.add_argument('--rolling', action='store_true', help='Are encodings rolling?')
parser.add_argument('--use-prior', action='store_true', help='Use a trainable prior for encoding?')
parser.add_argument('--anneal', action='store_true', help='Set learning rate annealing.')
parser.add_argument('--supervise', action='store_true', help='Set whether to use non-rollout supervision signal.')
parser.add_argument('--load-model', action='store_true', help='Whether to load model from checkpoint.')
parser.add_argument('--save-epochs', type=int, help='Save after every x epochs.')
parser.add_argument('--predict', type=str, help='Predict file.')
parser.add_argument('--checkpoint-dir', type=str, default='checkpoint/',
                    help='Checkpoint Dir.')
parser.add_argument('--settings', type=str,
                    help='Use a settings file.')

if '--settings' in sys.argv:
    settings_file = sys.argv[sys.argv.index('--settings')+1]
    with open(settings_file) as f:
        tokens = f.read().split()
    sys.argv = sys.argv[:1] + tokens + sys.argv[1:]

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.loss_fn = args.loss_fn.lower()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    num_devices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

MAX_MASS = 12.
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {'num_workers': 4}

train_set = PhysicsDataset(args.data_file, args.num_points, args.test_points, train=True)
train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_set = PhysicsDataset(args.data_file, args.num_points, args.test_points, train=False)
test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=args.batch_size, shuffle=False, **kwargs)

n_offsets = len(args.offsets)
n_objects = train_set.n_objects
assert args.enc_widths[-1] % n_objects == 0
enc_size = args.enc_widths[-1] / n_objects
assert args.code_size > enc_size

state_size = train_set.state_size
pred_model = networks.PredictNet(n_objects, args.code_size, n_offsets, args)
if args.cuda and num_devices > 1:
    pred_model = nn.DataParallel(pred_model, device_ids=range(num_devices))
state_to_code_model = networks.StateCodeModel(args.n_frames, state_size+enc_size, args.code_size, n_objects)
code_to_state_model = networks.MLP([args.code_size, args.n_frames*state_size])

enc_net = networks.ParallelEncNet(args.enc_widths, n_objects*state_size*args.n_enc_frames, args) 
enc_model = networks.ConfidenceWeightWrapper(enc_net, args)

if args.cuda:
    enc_net.cuda()
    enc_model.cuda()
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

lr_lambda = lambda epoch: args.lr_pred * max(0.03, math.e ** (-(epoch-1)*len(train_set)/args.alpha))

mse = nn.MSELoss()
l1 = nn.L1Loss()

""" HELPERS """

def zero_variable_(size, volatile=False):
    if args.cuda:
        return Variable(torch.cuda.FloatTensor(*size).zero_(), volatile=volatile)
    else:
        return Variable(torch.FloatTensor(*size).zero_(), volatile=volatile)

def log(*tokens):
    print(*tokens)
    text = " ".join([str(x) for x in tokens])
    if args.log:
        with open(log_file, 'a') as f:
            f.write(text + "\n")

def get_loss(pred, true, is_l1=False, discount=1.):
    # TODO this is a hack that presumes n_frames=2
    if is_l1:
        return discount * l1(pred, true)
    else:
        return discount * mse(pred, true)

def save_model():
    torch.save(pred_model.state_dict(),
               join(args.checkpoint_dir, 'pred_model.ckpt'))
    torch.save(state_to_code_model.state_dict(), 
               join(args.checkpoint_dir, 'state_to_code_model.ckpt'))
    torch.save(code_to_state_model.state_dict(), 
               join(args.checkpoint_dir, 'code_to_state_model.ckpt'))

def load_model():
    # TODO map_location is a hack to get cuda-saved tensors to load into non-cuda model.
    # But we can just move to cpu before saving in future.
    map_location = lambda storage, loc: storage
    pred_model.load_state_dict(torch.load(
        join(args.checkpoint_dir, 'pred_model.ckpt'), map_location=map_location))
    state_to_code_model.load_state_dict(torch.load(
        join(args.checkpoint_dir, 'state_to_code_model.ckpt'), map_location=map_location))
    code_to_state_model.load_state_dict(torch.load(
        join(args.checkpoint_dir, 'code_to_state_model.ckpt'), map_location=map_location))

def get_json_state(state):
    payload = []
    for sample in state:
        pos, vel = [], []
        for obj in sample:
            obj = obj.tolist()
            # TODO Hardcoded numbers. 
            pos.append({'x':obj[4], 'y':obj[5]})
            vel.append({'x':obj[6], 'y':obj[7]})

        payload.append({'pos':pos, 'vel':vel})

    return payload

""" TRAIN/TEST LOOPS """
def process_batch(x, train, discount=None, non_ro_weight=0., out_dir=None):
    if train:
        pred_optim.zero_grad()

    if args.cuda:
        x = [samp.cuda() for samp in x]
    x = [Variable(samp, volatile=not train) for samp in x]

    num_prep_frames = max(args.offsets)+args.n_frames-1
    enc_x, pred_x = x[:args.num_encode], x[args.num_encode - num_prep_frames:]

    mse_loss = zero_variable_((1,), volatile=not train)
    if not train:
        l1_loss = zero_variable_((1,), volatile=not train)
        base_l1_loss = zero_variable_((1,), volatile=not train)

    """Encoding step."""
    start_time = time.time()
    enc_x = [samp.view(args.batch_size, -1) for samp in enc_x]

    inps = []
    for start_ind in range(0, len(enc_x)-args.n_enc_frames+1):
        inp = enc_x[start_ind:start_ind+args.n_enc_frames]
        inp = torch.cat(inp, dim=1)
        inps.append(inp)
    enc = enc_model(inps)
    enc = enc.view(args.batch_size * n_objects, enc_size)

    """Prediction rollout step."""
    pred_x_wo_enc = [samp.view(-1, state_size) for samp in pred_x]
    pred_x = [torch.cat([samp, enc], dim=1) for samp in pred_x_wo_enc]

    aux_loss = 0
    # Add auxiliary losses
    codes = [None for _ in range(args.n_frames-1)]
    for start_ind in range(len(pred_x)-args.n_frames+1):
        inp = pred_x[start_ind:start_ind+args.n_frames]
        out = torch.cat(pred_x_wo_enc[start_ind:start_ind+args.n_frames], dim=1)
        code = state_to_code_model(inp) # TODO should we just cat inp before passing in?
        state = code_to_state_model(code)

        aux_loss_ = get_loss(state, out)
        mse_loss += aux_loss_
        aux_loss += aux_loss_.data[0]

        codes.append(code)

    ro_codes = [code for code in codes]

    # Add prediction losses
    num_preds = 0
    cur_discount = 1.
    preds, ro_preds, true_states = [], [], []
    for i in range(num_prep_frames, len(codes)):
        # i equals current frame index to predict.
        true_state = torch.cat(pred_x_wo_enc[i-1:i+1], dim=1)
        inps = [codes[i-offset] for offset in args.offsets]
        ro_inps = [ro_codes[i-offset] for offset in args.offsets]
        pred_code = pred_model(inps)
        ro_pred_code = pred_model(ro_inps)
        pred = code_to_state_model(pred_code)
        ro_pred = code_to_state_model(ro_pred_code)
        preds.append(pred)
        ro_preds.append(ro_pred)
        true_states.append(true_state)

        ro_codes[i] = ro_pred_code
        mse_loss += get_loss(ro_pred, true_state, discount=cur_discount)
        mse_loss += non_ro_weight * get_loss(pred, true_state)

        if not train:
            l1_loss += get_loss(ro_pred, true_state, is_l1=True)
            # TODO Hardcoded n-frames = 2
            base_pred = torch.cat([pred_x_wo_enc[i-1], pred_x_wo_enc[i-1]], dim=1)
            base_l1_loss += get_loss(base_pred, true_state, is_l1=True)
        num_preds += 1

        if discount:
            cur_discount *= discount

    """Save specific examples."""
    if out_dir:
        preds, true_states = [[a.data.numpy().reshape(-1, n_objects, args.n_frames*state_size)
                               for a in states] for states in (preds, true_states)]
        for i in range(len(preds[0])): # Iterate through each group in batch.
            pred = [a[i] for a in preds]
            true_state = [a[i] for a in true_states]
            log("Masses:", true_state[0][:,9] * MAX_MASS)
            with open(join(out_dir, 'data_'+str(i)+'.json'), 'w') as f:
                payload = {
                    'states': get_json_state(pred),
                    'true_states': get_json_state(true_state),
                }
                f.write(json.dumps(payload))

    if train:
        mse_loss.backward()
        pred_optim.step()

        return mse_loss.data[0], aux_loss
    else:
        return mse_loss.data[0], aux_loss, l1_loss.data[0], base_l1_loss.data[0], num_preds

def train_epoch(epoch):
    lr = lr_lambda(epoch) if args.anneal else args.lr_pred
    set_lr(lr)

    if args.discount:
        discount = 1 - math.e ** (-(epoch)*len(train_set)/float(args.beta))
    else:
        discount = None
    non_ro_weight = max(0., 2 * (1 - epoch / 40.)) if args.supervise else 0.

    log('Learning rate: {:.6f} Discount: {:.6f} Non-ro weight: {:.6f}'.format(lr,
        discount if discount is not None else 1., non_ro_weight))

    mse_loss, aux_loss, num_batches = 0, 0, 0
    start_time = time.time()

    for batch_idx, x in enumerate(train_loader):
        print("A", time.time() - start_time)
        mse_loss_, aux_loss_ = process_batch(x, train=True, discount=discount, non_ro_weight=non_ro_weight)
        mse_loss += mse_loss_
        aux_loss += aux_loss_
        num_batches += 1
        start_time = time.time()

    log('Time: {:.2f}s Epoch: {} Train Loss ({}): {:.5f} Aux {:.5f}'.format(
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

    log('Test Loss (L1): {:.5f} Base (L1): {:.5f} (MSE): {:.5f} Aux {:.5f}'.format(
        l1_loss / num_preds, base_l1_loss / num_preds, mse_loss / num_batches, aux_loss / num_batches))

def predict():
    batch = test_loader.__iter__().next()
    process_batch(batch, train=False, out_dir=args.predict)

if __name__ == '__main__':
    if args.log:
        log_file = join(args.log_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S.txt'))
        repo = git.Repo(search_parent_directories=True)
        log("Commit Hash:", repo.head.object.hexsha)

    log("Flags:", " ".join(sys.argv))

    if args.load_model:
        log("Load Model")
        load_model()

    if args.predict:
        log("Predict")
        predict()
    else:
        log("Start Training")
        for epoch in range(1, args.epochs+1):
            train_epoch(epoch)
            test_epoch(epoch)
            if args.save_epochs and epoch % args.save_epochs == 0:
                log("Saving Model")
                save_model()


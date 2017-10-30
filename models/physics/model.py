from __future__ import print_function

print("Importing")

import sys
sys.path.append('..')

from os.path import join
import os
import argparse
import glob
import time
from tqdm import tqdm
import math
import random
import json
import subprocess
import shutil
from datetime import datetime

import h5py
import git
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from loader import PhysicsDataset, EncPhysicsDataset, MAX_MASS
from parser import parser
import shared.networks as networks

""""" CONFIG """""

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.rolling = False
args.enc = args.enc or args.all
args.pred = args.pred or args.all

start_epoch = 1

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    args.num_devices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if args.num_devices > 1:
        args.batch_size = (args.batch_size // args.num_devices) * args.num_devices
        args.num_points = (args.num_points // args.num_devices) * args.num_devices
        args.test_points = (args.test_points // args.num_devices) * args.num_devices
else:
    args.num_devices = 1

if args.progbar:
    progbar = lambda x: tqdm(x, leave=False)
else:
    progbar = lambda x: x

""""" INITIALIZATIONS """""

print("Initializing")

""" DATA LOADERS """
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {'num_workers': 4}

train_set = PhysicsDataset(args.data_file, args.num_points, args.test_points, train=True)
test_set = PhysicsDataset(args.data_file, args.num_points, args.test_points, train=False)

train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=args.batch_size, shuffle=False, **kwargs)

n_offsets = len(args.offsets)
n_objects = train_set.n_objects

state_size = train_set.state_size
y_size = train_set.y_size

mse = nn.MSELoss()
l1 = nn.L1Loss()

""" Encoding Models """

if args.enc:
    if args.enc_model == 'lstm':
        assert len(args.enc_widths) == 1
        assert args.trans_widths[-1] % n_objects == 0
        enc0_structure = [(args.enc_widths[-1], args.enc_widths[-1])] * args.depth
        enc_net = networks.LSTMEncNet(args.enc_widths[-1],
                                      n_objects*state_size*args.n_enc_frames,
                                      depth=args.depth)
        enc_model = networks.RecurrentWrapper(enc_net, args, enc0_structure)
        enc_size = args.trans_widths[-1] // n_objects
        trans_model = networks.TransformNet(args.enc_widths[-1],
                                            args.trans_widths[:-1],
                                            args.trans_widths[-1])
        # enc_model = networks.SymEncWrapper(n_objects, state_size, args)
        # enc_size = args.trans_widths[-1]
        if not args.pred:
            assert enc_size == y_size
    elif args.enc_model == 'parallel':
        raise RuntimeError("Currently Unsupported")
    else:
        raise RuntimeError("Model type {} not recognized.".format(args.enc_model))

    if args.cuda:
        enc_net.cuda()
        enc_model.cuda()
        trans_model.cuda()

    enc_optim = optim.Adam(
            list(enc_model.parameters()) + list(trans_model.parameters()),
            lr=args.lr_enc)
else:
    enc_size = y_size

""" Prediction Models """
if args.pred:
    pred_model = networks.PredictNet(n_objects, args.code_size, n_offsets, args)
    state_to_code_model = networks.StateCodeModel(args.n_frames, state_size+enc_size,
            args.code_size, n_objects)
    code_to_state_model = networks.MLP([args.code_size, state_size])

    if args.cuda:
        pred_model.cuda()
        state_to_code_model.cuda()
        code_to_state_model.cuda()

    pred_optim = optim.Adam(
            [x for x in list(pred_model.parameters()) if x.requires_grad] +
            list(state_to_code_model.parameters()) +
            list(code_to_state_model.parameters()),
            lr=args.lr_pred)

""""" LOGGING SETUP """""

if args.log_dir == '':
    log_counter_file = join(args.base_log_folder, 'log_counter.txt')
    with open(log_counter_file, 'r') as f:
        new_counter = int(f.read()) + 1

    with open(log_counter_file, 'w') as f:
        f.write(str(new_counter) + '\n')

    log_folder = join(args.base_log_folder, str(new_counter))
else:
    log_folder = join(args.base_log_folder, args.log_dir)

log_file = join(log_folder, 'log.txt')

print("Logging to file {}".format(log_file))

if os.path.exists(log_file):
    if not args.continue_train and \
            raw_input("Log file already exists, overwrite? (y/n) ") != 'y':
        sys.exit(0)
else:
    os.makedirs(log_folder)

""""" HELPERS """""

def open_encs_file():
    global encs_file
    encs_file = h5py.File(join(log_folder, 'encs.h5'), 'a')
    if 'enc' not in encs_file:
        encs_file.create_dataset('enc', (len(train_set), n_objects, enc_size), dtype='f')
    if 'y' not in encs_file:
        encs_file.create_dataset('y', (len(train_set), n_objects, y_size), dtype='f')

    return encs_file

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

def save_checkpoint(epoch):
    state = {'epoch': epoch}

    if args.enc:
        state['enc_state_dict'] = enc_model.state_dict()
        state['trans_state_dict'] = trans_model.state_dict()
        state['enc_optim'] = enc_optim.state_dict()
    if args.pred:
        state['pred_state_dict'] = pred_model.state_dict()
        state['cts_state_dict'] = code_to_state_model.state_dict()
        state['stc_state_dict'] = state_to_code_model.state_dict()
        state['pred_optim'] = pred_optim.state_dict()

    filename = '.epoch{}.tar'.format(epoch)
    filename = join(log_folder, filename)
    torch.save(state, filename)
    latest_filename = join(log_folder, 'latest.tar')
    shutil.copyfile(filename, latest_filename)

# Loads from latest.tar
def continue_checkpoint():
    filename = join(log_folder, 'latest.tar')
    if not os.path.exists(filename):
        raise RuntimeError("No checkpoint found at {}".format(filename))
    checkpoint = torch.load(filename)
    global start_epoch
    start_epoch = checkpoint['epoch'] + 1

    for key in checkpoint:
        if isinstance(checkpoint[key], dict):
            checkpoint[key] = {
                k[7:] if 'module' in k else k:v
                for k, v in checkpoint[key].items()
            }

    if args.enc:
        enc_model.load_state_dict(checkpoint['enc_state_dict'])
        trans_model.load_state_dict(checkpoint['trans_state_dict'])
        enc_optim.load_state_dict(checkpoint['enc_optim'])

    if args.pred:
        pred_model.load_state_dict(checkpoint['pred_state_dict'])
        code_to_state_model.load_state_dict(checkpoint['cts_state_dict'])
        state_to_code_model.load_state_dict(checkpoint['stc_state_dict'])
        pred_optim.load_state_dict(checkpoint['pred_optim'])

def zero_variable_(size, volatile=False):
    if args.cuda:
        return Variable(torch.cuda.FloatTensor(*size).zero_(), volatile=volatile)
    else:
        return Variable(torch.FloatTensor(*size).zero_(), volatile=volatile)

def normal_variable_(size, std=1.0, volatile=False):
    if args.cuda:
        return Variable(torch.cuda.FloatTensor(*size).normal_(std=std), volatile=volatile)
    else:
        return Variable(torch.FloatTensor(*size).normal_(std=std), volatile=volatile)

def log(*tokens):
    print(*tokens)
    text = " ".join([str(x) for x in tokens])
    with open(log_file, 'a') as f:
        f.write(text + "\n")

def get_loss(pred, true, loss_fn='mse', discount=1., pos_only=False, vel_only=False):
    if pos_only:
        pred = pred[:,:2]
        true = true[:,:2]
    elif vel_only:
        pred = pred[:,2:]
        true = true[:,2:]

    if loss_fn == 'mse':
        return discount * mse(pred, true)
    elif loss_fn == 'l1':
        return discount * l1(pred, true)
    else:
        raise RuntimeError("Currently not supported.")

def get_json_state(state):
    payload = []
    for sample in state:
        pos, vel = [], []
        for obj in sample:
            obj = obj.tolist()
            # TODO Hardcoded numbers. 
            pos.append({'x':obj[0], 'y':obj[1]})
            vel.append({'x':obj[2], 'y':obj[3]})

        payload.append({'pos':pos, 'vel':vel})

    return payload

def get_ro_discount(epoch):
    if args.beta:
        return 1 - math.e ** (-float(epoch-1) / args.beta)
    return 1


save_enc_ind = 0
def save_encs(enc, y):
    global save_enc_ind, encs_file
    enc = enc.data.cpu().numpy().reshape(-1, n_objects, enc_size)
    y = y.data.cpu().numpy().reshape(-1, n_objects, y_size)
    num_points = len(enc)
    encs_file['enc'][save_enc_ind:save_enc_ind+num_points] = enc
    encs_file['y'][save_enc_ind:save_enc_ind+num_points] = y

    save_enc_ind = (save_enc_ind + num_points) % len(train_set)

    if save_enc_ind == 0:
        print('Saving Encs')
        encs_file.close()
        open_encs_file()

""" TRAIN/TEST LOOPS """
def process_batch(enc_x, pred_xs, y, train, non_ro_weight=0., render=False, ro_discount=1.):
    if train:
        if args.enc:
            enc_optim.zero_grad()
        if args.pred:
            pred_optim.zero_grad()

    if args.cuda:
        enc_x = enc_x.cuda()
        pred_xs = pred_xs.cuda()
        y = y.cuda()

    enc_x = list(enc_x.permute(1, 0, 2, 3).contiguous()) # timestep dimension first
    pred_xs = [list(run) for run in pred_xs.permute(1, 2, 0, 3, 4).contiguous()] # runs and timestep dimensions first

    enc_x = [Variable(samp, volatile=not train) for samp in enc_x]
    pred_xs = [[Variable(samp, volatile=not train) for samp in pred_x] for pred_x in pred_xs]
    y = Variable(y).view(-1, y_size)

    num_prep_frames = max(args.offsets)+args.n_frames-1

    if args.max_pred_frames and not render:
        pred_xs = [pred_x[:args.max_pred_frames+num_prep_frames] for pred_x in pred_xs]

    loss = zero_variable_((1,), volatile=not train)
    aux_loss = zero_variable_((1,), volatile=not train)
    if not train:
        pred_loss = zero_variable_((1,), volatile=True)
        base_pred_loss = zero_variable_((1,), volatile=True)
        base_discount_pred_loss = zero_variable_((1,), volatile=True)

    """Encoding step."""
    if args.enc:
        enc_x = [samp.view(-1, n_objects * state_size) for samp in enc_x]

        inps = []
        for start_ind in range(0, len(enc_x)-args.n_enc_frames+1):
            inp = enc_x[start_ind:start_ind+args.n_enc_frames]
            inp = torch.cat(inp, dim=1)
            inps.append(inp)

        enc = enc_model(inps)
        enc = trans_model(enc)
        enc = enc.view(-1, enc_size)

        if args.save_encs and train:
            save_encs(enc, y)

        if not args.pred:
            l1_loss = 0
            enc = enc.view(-1, enc_size * n_objects)
            loss += get_loss(enc, y, loss_fn=args.loss_fn)
            if not train:
                l1_loss += get_loss(enc, y, loss_fn='l1')

            if train:
                loss.backward()
                enc_optim.step()
                return (loss.data[0], 0, 0)
            else:
                return (loss.data[0], 0, 0, 0, 0, 0)
    elif args.pred:
        if args.no_enc:
            enc = zero_variable_(y.size(), volatile=not train)
        else:
            enc = y

    """Prediction rollout step."""
    if args.pred:
        for pred_x in pred_xs:
            non_ro_loss = zero_variable_((1,), volatile=not train)

            pred_x_wo_enc = [samp.view(-1, state_size) for samp in pred_x]
            pred_x = [torch.cat([samp, enc], dim=1) for samp in pred_x_wo_enc]

            # Add auxiliary losses
            codes = [None for _ in range(args.n_frames-1)]
            for start_ind in range(len(pred_x)-args.n_frames+1):
                inp = pred_x[start_ind:start_ind+args.n_frames]
                code = state_to_code_model(inp) # TODO should we just cat inp before passing in?

                if not args.predict_delta:
                    out = pred_x_wo_enc[start_ind+args.n_frames-1]
                    state = code_to_state_model(code)
                    aux_loss_ = get_loss(state, out, loss_fn=args.loss_fn)
                    loss += aux_loss_
                    aux_loss += aux_loss_

                codes.append(code)

            ro_codes = [code for code in codes]

            # Add prediction losses
            num_preds = 0
            preds = [x for x in pred_x_wo_enc]
            ro_preds = [x for x in pred_x_wo_enc]
            true_states = [x for x in pred_x_wo_enc]
            cur_discount = 1.
            for i in range(num_prep_frames, len(codes)):
                # i equals current frame index to predict.
                inps = torch.stack([codes[i-offset] for offset in args.offsets], dim=1)
                pred_code = pred_model(inps)
                pred = code_to_state_model(pred_code)

                ro_inps = torch.stack([ro_codes[i-offset] for offset in args.offsets], dim=1)
                ro_pred_code = pred_model(ro_inps)
                ro_pred = code_to_state_model(ro_pred_code)

                if args.predict_delta:
                    ro_pred += pred_x_wo_enc[i-1] if len(preds) == 0 else ro_preds[-1]
                    pred += pred_x_wo_enc[i-1]

                preds[i] = pred
                ro_preds[i] = ro_pred

                if args.recode:
                    inps = ro_preds[i-args.n_frames+1:i+1]
                    inps = [torch.cat([samp, enc], dim=1) for samp in inps]
                    ro_pred_code = state_to_code_model(inps)

                if args.noise and train:
                    ro_codes[i] = ro_pred_code + normal_variable_(ro_pred_code.size(), std=args.noise)
                else:
                    ro_codes[i] = ro_pred_code

                true_state = pred_x_wo_enc[i]

                loss += get_loss(ro_pred, true_state, loss_fn=args.loss_fn) * cur_discount
                non_ro_loss += get_loss(pred, true_state, loss_fn=args.loss_fn)

                if not train:
                    pred_loss += get_loss(ro_pred, true_state, loss_fn='mse', pos_only=True)
                    base_pred_loss += get_loss(pred_x_wo_enc[i-1], true_state, loss_fn='mse', pos_only=True)
                    base_discount_pred_loss += get_loss(pred_x_wo_enc[i-1], true_state, loss_fn='mse', pos_only=True) * cur_discount

                cur_discount *= ro_discount
                num_preds += 1

        if render:
            ro_preds, true_states = [[a.data.cpu().numpy().reshape(-1, n_objects, state_size)
                                   for a in states] for states in (ro_preds, true_states)]
            for i in range(len(ro_preds[0])): # Iterate through each group in batch.
                ro_pred = [a[i] for a in ro_preds]
                true_state = [a[i] for a in true_states]
                with open(join(log_folder, 'data_'+str(i)+'.json'), 'w') as f:
                    payload = {
                        'states': get_json_state(ro_pred),
                        'true_states': get_json_state(true_state),
                    }
                    f.write(json.dumps(payload))

        """Apply gradients."""
        if args.non_rollout:
            loss = aux_loss + non_ro_loss
        else:
            loss += non_ro_weight * non_ro_loss

    if train:
        loss.backward()
        if args.enc:
            enc_optim.step()
        if args.pred:
            pred_optim.step()

        return loss.data[0], non_ro_loss.data[0], aux_loss.data[0]
    else:
        return (loss.data[0], non_ro_loss.data[0], aux_loss.data[0],
                pred_loss.data[0], base_pred_loss.data[0], base_discount_pred_loss.data[0], 
                num_preds)

def train_epoch(epoch):
    start_time = time.time()

    ro_discount = get_ro_discount(epoch)
    log('Log Folder: {:s} Non-ro weight: {:.6f} RO discount: {:.6f}'.format(log_folder, args.non_ro_weight, ro_discount))

    loss, non_ro_loss, aux_loss, num_batches = 0, 0, 0, 0

    for batch_idx, (enc_x, pred_xs, y) in enumerate(progbar(train_loader)):
        loss_, non_ro_loss_, aux_loss_ = \
            process_batch(enc_x, pred_xs, y, train=True, non_ro_weight=args.non_ro_weight,
                          ro_discount=ro_discount)
        non_ro_loss += non_ro_loss_
        loss += loss_
        aux_loss += aux_loss_
        num_batches += 1

    log('TRAIN: Epoch: {} Total Loss: {:.5f} Aux Loss: {:.5f} Non-ro Loss: {:.5f}'
        ' Time: {:.2f}s '.format(
        epoch, loss / num_batches, aux_loss / num_batches, non_ro_loss / num_batches,
        time.time() - start_time))

def test_epoch(epoch):
    ro_discount = get_ro_discount(epoch)
    loss, non_ro_loss, aux_loss, pred_loss, base_pred_loss, base_discount_pred_loss, num_batches, num_preds = 0, 0, 0, 0, 0, 0, 0, 0

    for batch_idx, (enc_x, pred_xs, y) in enumerate(test_loader):
        loss_, non_ro_loss_, aux_loss_, pred_loss_, base_pred_loss_, base_discount_pred_loss_, num_preds_ = \
            process_batch(enc_x, pred_xs, y, train=False, non_ro_weight=args.non_ro_weight,
                          ro_discount=ro_discount)
        num_preds += num_preds_
        loss += loss_
        non_ro_loss += non_ro_loss_
        aux_loss += aux_loss_
        pred_loss += pred_loss_
        base_pred_loss += base_pred_loss_
        base_discount_pred_loss += base_discount_pred_loss_
        num_batches += 1

    if args.render:
        process_batch(test_set.enc_x_long, test_set.pred_xs_long, test_set.y_long,
                      train=False, render=True)

    log('TEST: Total Loss: {:.5f} Pred Loss: {:.5f} Aux Loss: {:.5f} Non-ro Loss: {:.5f}'
        ' Base Loss: {:.5f} Base Discount Loss: {:.5f}'.format(
        loss / num_batches, pred_loss / num_batches, aux_loss / num_batches,
        non_ro_loss / num_batches, base_pred_loss / num_batches, base_discount_pred_loss / num_batches))

if __name__ == '__main__':
    if args.continue_train:
        log("Continuing train...")
        continue_checkpoint()
    log("{}".format(sys.argv))
    if args.enc:
        if args.num_devices > 1:
            enc_model = nn.DataParallel(enc_model)
            trans_model = nn.DataParallel(trans_model)
        log("Enc net params: {}".format(networks.num_params(enc_model)))
        log("Trans net params: {}".format(networks.num_params(trans_model)))
    if args.pred:
        if args.num_devices > 1:
            pred_model = nn.DataParallel(pred_model)
            state_to_code_model = nn.DataParallel(state_to_code_model)
            code_to_state_model = nn.DataParallel(code_to_state_model)
        log("Pred net params: {}".format(networks.num_params(pred_model)))
    log("Git commit: {}".format(subprocess.check_output(['git', 'rev-parse', 'HEAD'])[:-1]))

    if args.enc and args.save_encs:
        open_encs_file()

    print("Start Training")
    for epoch in range(start_epoch, args.epochs+start_epoch):
        train_epoch(epoch)
        save_checkpoint(epoch)
        test_epoch(epoch)
        log('')


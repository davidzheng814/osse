from os.path import join
import os
import argparse
import glob
import time
import math
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

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
parser.add_argument('--data-dir', type=str, default=join(DATAROOT, '.physics_n3_t60_f120'),
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
parser.add_argument('--earliest-rollout', action="store_true",
                    help="Set rollout index to earliest rollout")
parser.add_argument('--discount', action="store_true",
                    help="Whether to discount future rollout states at first.")
parser.add_argument('---beta', type=float, default=1.5e5,
                    help="The rollout discount exponent. See code.")
parser.add_argument('---noise', type=float,
                    help="The rollout discount exponent. See code.")
parser.add_argument('--predict-ind', type=int, default=0,
                    help="First sample used as prediction.")
parser.add_argument('--max-samples', type=int,
                    help='max samples per group.')
parser.add_argument('--anneal', action='store_true', help='Set learning rate annealing.')
parser.add_argument('--supervise', action='store_true', help='Set whether to use non-rollout supervision signal.')
parser.add_argument('--load-model', action='store_true', help='Whether to load model from checkpoint.')
parser.add_argument('--save-epochs', type=int, help='Save after every x epochs.')
parser.add_argument('--predict', type=str, help='Predict file.')
parser.add_argument('--checkpoint-dir', type=str, default='checkpoint/',
                    help='Checkpoint Dir.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.loss_fn = args.loss_fn.lower()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    num_devices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

""" DATA LOADERS """

MAX_MASS = 12.
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
        # TODO Hardcoding max mass.
        masses = np.tile(masses / MAX_MASS, (x.shape[0], 1))
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
                rand_ind = random.randint(0, len(x)-args.max_samples)
                x = x[rand_ind:rand_ind+args.max_samples]

        return x


class MLP(nn.Module):
    def __init__(self, widths, reshape=False, tanh=False, relu=True, n_objects=None):
        """ Only set n_objects if reshape = True"""
        super(MLP, self).__init__()

        self.reshape = reshape
        self.relu = relu
        self.tanh = tanh
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
            if i != len(self.linears) - 1:
                if self.relu:
                    h = F.relu(h)
                elif self.tanh:
                    h = F.tanh(h)

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
        # self.linear = nn.Linear(n_frames*state_size, code_size)
        self.mlp = MLP([n_frames*state_size, code_size])

    def forward(self, x):
        # x = [self.linear(inp) for inp in x]
        h = torch.cat(x, dim=1)
        # h = self.linear(h)
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

n_offsets = len(args.offsets)
n_objects = train_set.n_objects
state_size = train_set.state_size
pred_model = PredictNet(n_objects, args.code_size, n_offsets)
if args.cuda and num_devices > 1:
    pred_model = nn.DataParallel(pred_model, device_ids=range(num_devices))
state_to_code_model = StateCodeModel(args.n_frames, state_size, args.code_size, n_objects)
code_to_state_model = MLP([args.code_size, args.n_frames*state_size])

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

lr_lambda = lambda epoch: args.lr_pred * max(0.03, math.e ** (-(epoch-1)*len(train_set)/args.alpha))

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

def get_loss(pred, true, is_l1=False, discount=1.):
    # TODO this is a hack that presumes n_frames=2
    pred = torch.cat([pred[:,:state_size-1],
                      pred[:,state_size:2*state_size-1]], dim=1)
    true = torch.cat([true[:,:state_size-1],
                      true[:,state_size:2*state_size-1]], dim=1)
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
            pos.append({'x':obj[5], 'y':obj[6]})
            vel.append({'x':obj[7], 'y':obj[8]})

        payload.append({'pos':pos, 'vel':vel})

    return payload

""" TRAIN/TEST LOOPS """
def process_batch(x, train, discount=None, non_ro_weight=0., out_dir=None):
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
        out = torch.cat(inp, dim=1)
        code = state_to_code_model(inp)
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
    for i in range(max(args.offsets)+args.n_frames-1, len(codes)):
        # i equals current frame index to predict.
        true_state = torch.cat(x[i-1:i+1], dim=1)
        inps = [codes[i-offset] for offset in args.offsets]
        ro_inps = [ro_codes[i-offset] for offset in args.offsets]
        pred_code = pred_model(inps)
        ro_pred_code = pred_model(ro_inps)
        pred = code_to_state_model(pred_code)
        ro_pred = code_to_state_model(ro_pred_code)
        preds.append(pred)
        ro_preds.append(ro_pred)
        true_states.append(true_state)

        is_rollout = args.earliest_rollout or (args.rollout_ind and i >= args.rollout_ind)
        if is_rollout:
            ro_codes[i] = ro_pred_code
            mse_loss += get_loss(ro_pred, true_state, discount=cur_discount)
            mse_loss += non_ro_weight * get_loss(pred, true_state)
        else:
            mse_loss += get_loss(pred, true_state)

        if not train:
            l1_loss += get_loss(ro_pred, true_state, is_l1=True)
            # TODO Hardcoded n-frames = 2
            base_pred = torch.cat([x[i-1], x[i-1]], dim=1)
            base_l1_loss += get_loss(base_pred, true_state, is_l1=True)
        num_preds += 1

        if is_rollout and discount:
            cur_discount *= discount

    # Write predictions to json file. 
    if out_dir:
        preds, true_states = [[x.data.numpy().reshape(-1, n_objects, args.n_frames*state_size)
                               for x in states] for states in (preds, true_states)]
        for i in range(len(preds[0])): # Iterate through each group in batch.
            pred = [x[i] for x in preds]
            true_state = [x[i] for x in true_states]
            print("Masses:", true_state[0][:,9] * MAX_MASS)
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
        mse_loss_, aux_loss_ = process_batch(x, train=True, discount=discount, non_ro_weight=non_ro_weight)
        mse_loss += mse_loss_
        aux_loss += aux_loss_
        num_batches += 1

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


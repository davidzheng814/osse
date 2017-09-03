from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import time

import util

# Number of trainable scalar parameters in model
def num_params(model):
    parameters = list(model.parameters())
    sizes = [reduce(mul, p.size(), 1) for p in parameters if p.requires_grad]
    return sum(sizes)

def get_wrapper(wrapper, encoder, args):
    wrappermap = {'confweight': ConfidenceWeightWrapper,
            'recurrent': RecurrentWrapper}
    if wrapper in wrappermap:
        return wrappermap[wrapper](encoder, args)
    else:
        raise RuntimeError("Wrapper " + wrapper + " not found!")

def get_encoder(model, args, x_size, y_size):
    if model == 'rnn':
        return BasicEncNet(args.enc_widths[-1], x_size + y_size, use_lstm=False)
    elif model == 'lstm':
        return BasicEncNet(args.enc_widths[-1], x_size + y_size, use_lstm=True)
    elif model == 'parallel':
        return ParallelEncNet(args.enc_widths, x_size + y_size)
    raise RuntimeError("Encoder " + model + " not found!")

def get_predictor(model, args, x_size, y_size):
    if model == 'basic':
        return BasicPredictNet(args.enc_widths[-1], x_size, y_size)
    if model == 'nonlinear':
        return NLPredictNet(args.enc_widths[-1], args.pred_widths, x_size, y_size)
    raise RuntimeError("Predictor " + model + " not found!")

""" MODEL WRAPPERS """

class BaseWrapper(nn.Module):
    def __init__(self):
        super(BaseWrapper, self).__init__()
        self.rolling = False
    
    def is_rolling(self):
        return self.rolling

class RecurrentWrapper(BaseWrapper):
    """
    Wraps a model that outputs an encoding at every time step after receiving the
    previous encoding.
    """
    def __init__(self, step_model, args, enc0_structure=None):
        super(RecurrentWrapper, self).__init__()
        self.step_model = step_model
        self.use_cuda = args.cuda
        self.enc_width = args.enc_widths[-1]
        if enc0_structure is None:
            self.enc0 = nn.Parameter(torch.Tensor(np.zeros((self.enc_width,))).cuda())
        else:
            def initialize_structure(enc0_structure):
                li = [nn.Parameter(torch.Tensor(np.zeros((x,))).cuda())
                      if type(x) is int else initialize_structure(x) 
                      for x in enc0_structure]
                return li
            self.enc0 = nn.Parameter(torch.Tensor(np.zeros((self.enc_width,))).cuda()),\
                initialize_structure(enc0_structure)

        self.rolling = args.rolling

    # sample_batch has shape (time_steps, batch_size, x_size)
    # only the last 2 layers are torch Tensors
    def forward(self, sample_batch):
        batch_size = sample_batch[0].size()[0] # get dynamic batch size
        def batch_enc_structure(enc):
            if type(enc) is list:
                return [batch_enc_structure(x) for x in enc]
            elif type(enc) is tuple:
                return tuple(batch_enc_structure(x) for x in enc)
            else:
                return enc.repeat(batch_size, 1)
        enc = batch_enc_structure(self.enc0)
        encs = []
        encs.append(enc)

        for i, sample in enumerate(sample_batch[:-1]):
            if self.use_cuda:
                sample = sample.cuda()
            if not isinstance(sample, Variable):
                sample = Variable(sample)
            enc = self.step_model(sample, enc)
            if type(enc) is tuple:
                # For LSTM, first element of model output is output encoding
                encs.append(enc[0])
            else:
                encs.append(enc)

        if self.rolling:
            return encs
        else:
            return encs[-1]

class ConfidenceWeightWrapper(BaseWrapper):
    """
    Wraps a model that outputs a local encoding and confidence for each timestep.
    """
    def __init__(self, step_model, args):
        super(ConfidenceWeightWrapper, self).__init__()
        self.step_model = step_model
        self.enc_width = args.enc_widths[-1]
        self.use_cuda = args.cuda
        self.use_prior = args.use_prior
        self.rolling = args.rolling

    # sample_batch has shape (time_steps, batch_size, x_size)
    # only the last 2 layers are torch Tensor
    def forward(self, sample_batch):
        batch_size = sample_batch[0].size()[0] # get dynamic batch size
        n_time_steps = len(sample_batch)
        w_encs = util.zero_variable((n_time_steps, batch_size, self.enc_width),
                self.use_cuda)
        confs = util.zero_variable((n_time_steps, batch_size, self.enc_width),
                self.use_cuda)
        enc0, conf0 = self.step_model.get_enc_conf0()
        if self.use_prior:
            enc0_expand = enc0.repeat(batch_size,1)
            conf0_expand = conf0.repeat(batch_size,1)
            w_encs[0,:,:] = enc0_expand * conf0_expand
            confs[0,:,:] = conf0_expand
        else:
            confs[0,:,:] = 1e-9 # Prevents nan in initial backprop

        for i, sample in enumerate(sample_batch[:-1]):
            if self.use_cuda:
                sample = sample.cuda()
            if not isinstance(sample, Variable):
                sample = Variable(sample)
            local_enc, conf = self.step_model(sample)
            w_encs[i+1] = local_enc * conf
            confs[i+1] = conf
        encs = torch.cumsum(w_encs, dim=0) / torch.cumsum(confs, dim=0)
        encs = list(encs)
        if self.rolling:
            return encs
        else:
            return encs[-1]

""" PredNets """

class BasicPredictNet(nn.Module):
    def __init__(self, enc_size, x_size, y_size):
        super(BasicPredictNet, self).__init__()
        self.lin = nn.Linear(enc_size, x_size * y_size)
        self.x_size, self.y_size = x_size, y_size

    def forward(self, x, enc):
        x = x.unsqueeze(1)
        W = self.lin(enc).view([-1, self.x_size, self.y_size])
        h = torch.bmm(x, W).squeeze()
        return h, W

class NLPredictNet(nn.Module):
    '''
    A predict network that applies nonlinear layers onto the encoding to form the W
    matrix in a hardcoded y = W * x output.
    '''
    def __init__(self, enc_width, pred_widths, x_size, y_size):
        super(NLPredictNet, self).__init__()
        model = []
        layer_sizes = [enc_width] + pred_widths
        for in_width, out_width in zip(layer_sizes[:-1], layer_sizes[1:]):
            model.append(nn.Linear(in_width, out_width))
            model.append(nn.ReLU())
        model.append(nn.Linear(pred_widths[-1], x_size * y_size))
        self.model = nn.Sequential(*model)
        self.x_size, self.y_size = x_size, y_size

    def forward(self, x, enc):
        x = x.unsqueeze(1)
        W = self.model(enc).view([-1, self.x_size, self.y_size])
        h = torch.bmm(x, W).squeeze()
        return h, W

""" TransformNets """

class TransformNet(nn.Module):
    '''
    Transforms an encoding directly into a prediction with no additional input.
    '''
    def __init__(self, enc_size, widths, y_size):
        super(TransformNet, self).__init__()
        layer_sizes = [enc_size] + widths
        model = []
        for in_width, out_width in zip(layer_sizes[:-1], layer_sizes[1:]):
            model += [nn.Linear(in_width, out_width)]
            model += [nn.ReLU()]
        model.append(nn.Linear(widths[-1], y_size))
        self.model = nn.Sequential(*model)

    def forward(self, enc):
        return self.model(enc)

""" EncNets """

class BasicEncNet(nn.Module):
    def __init__(self, enc_size, sample_size):
        super(BasicEncNet, self).__init__()
        self.use_lstm = use_lstm

        enc_weights = [enc_size] * 3
        enc_weights = [sample_size+enc_size] + enc_weights
        # inp_weights = [x_size+y_size] + [enc_size] * 4

        self.enc_linears = nn.ModuleList([
            nn.Linear(inp, out) for inp, out in zip(enc_weights[:-1], enc_weights[1:])])

    def forward(self, sample, enc0):
        h = torch.cat((sample, enc0), dim=1)
        for i, lin in enumerate(self.enc_linears):
            if i == len(self.enc_linears) - 1: 
                h = lin(h)
            else:
                h = F.relu(lin(h))

        return h

class LSTMEncNet(nn.Module):
    def __init__(self, enc_size, sample_size, depth=1):
        super(LSTMEncNet, self).__init__()

        assert depth >= 1
        lstms = []
        lstms += [nn.LSTMCell(sample_size, enc_size)]
        for _ in xrange(depth - 1):
            lstms += [nn.LSTMCell(enc_size, enc_size)]
        self.lstms = nn.ModuleList(lstms)

    def forward(self, sample, enc0):
        _, states = enc0
        h = sample
        new_states = []
        for i in xrange(len(self.lstms)):
            prev_h, prev_c = states[i]
            h, c = self.lstms[i](h, (prev_h, prev_c))
            new_states.append((h,c))
        return h, new_states

class ParallelEncNet(nn.Module):
    def __init__(self, widths, sample_size, args):
        super(ParallelEncNet, self).__init__()

        self.enc0 = torch.Tensor(np.zeros((1, widths[-1])))
        self.conf0 =torch.Tensor(np.ones((1, widths[-1])))
        if args.cuda:
            self.enc0.cuda()
            self.conf0.cuda()
        self.enc0 = nn.Parameter(self.enc0)
        self.conf0 = nn.Parameter(self.conf0)

        enc_weights = [sample_size] + widths

        self.enc_linears = nn.ModuleList([
                nn.Linear(inp, out) for inp, out in zip(enc_weights[:-1], enc_weights[1:])])
        self.conf_linears = nn.ModuleList([
                nn.Linear(inp, out) for inp, out in zip(enc_weights[:-1], enc_weights[1:])])

    def get_enc_conf0(self):
        return self.enc0, self.conf0

    def forward(self, sample):
        h = sample
        for i, lin in enumerate(self.enc_linears):
            if i == len(self.enc_linears) - 1: 
                h = lin(h)
            else:
                h = F.relu(lin(h))

        conf = sample
        for i, lin in enumerate(self.conf_linears):
            if i == len(self.conf_linears) - 1: 
                conf = torch.sigmoid(lin(conf))
            else:
                conf = F.relu(lin(conf))

        return h, conf

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
    def __init__(self, n_objects, code_size, args):
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

        self.index = index

    def forward(self, x):
        base = x.repeat(1, self.n_objects-1, 1)

        index = Variable(torch.LongTensor(self.index).cuda())
        scrambled = torch.index_select(base, 1, index)

        h = torch.cat([base, scrambled], dim=2)
        h = h.view(-1, self.inp_size)
        h = self.MLP(h)
        h = h.view(-1, self.n_objects-1, self.n_objects, self.code_size)
        h = torch.sum(h, 1).view(-1, self.code_size)

        return h

class InteractionNet(nn.Module):
    def __init__(self, n_objects, code_size, args):
        super(InteractionNet, self).__init__()

        self.n_objects = n_objects
        self.code_size = code_size

        self.re_net = RelationNet(n_objects, code_size, args)
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
    def __init__(self, n_objects, code_size, num_offsets, args):
        super(PredictNet, self).__init__()

        self.n_objects = n_objects
        self.code_size = code_size
        self.num_offsets = num_offsets

        self.inets = nn.ModuleList([
            InteractionNet(n_objects, code_size, args) for _ in range(num_offsets)])

        self.agg = MLP([num_offsets * code_size, code_size, code_size])

    def forward(self, inps):
        preds = []
        count = 0
        start_time = time.time()
        for i, inet in enumerate(self.inets):
            inp = inps[:,i].contiguous()
            preds.append(inet(inp))
            count += 1

        h = torch.cat(preds, dim=1)

        h = self.agg(h)

        return h


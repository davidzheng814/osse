import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

def get_encoder(model, args, x_size, y_size):
    if model == 'ff':
        return BasicEncNet(args.enc_widths[-1], x_size, y_size, use_lstm=False)
    elif model == 'lstm':
        return BasicEncNet(args.enc_widths[-1], x_size, y_size, use_lstm=True)
    elif model == 'parallel':
        return ParallelEncNet(args.enc_widths, x_size, y_size)
    raise RuntimeError("Encoder " + model + " not found!")

def get_predictor(model, args, x_size, y_size):
    if model == 'basic':
        return BasicPredictNet(args.enc_widths[-1], x_size, y_size)
    if model == 'nonlinear':
        return NLPredictNet(args.enc_widths[-1], args.pred_widths, x_size, y_size)
    raise RuntimeError("Predictor " + model + " not found!")

""" MODEL WRAPPERS """

class ConfidenceWeightWrapper(nn.Module):
    """
    Wraps a model that outputs a weight and confidence for a given timestep
    to produce an overall encoding from a sequence of time steps.
    """
    def __init__(self, step_model, use_cuda=True, use_prior=False):
        super(ConfidenceWeightWrapper, self).__init__()
        self.step_model = step_model
        self.use_cuda = use_cuda
        self.use_prior = use_prior

    # x_batch has shape (time_steps, batch_size, x_size)
    # only the last 2 layers are torch Tensors
    def forward(self, x_batch, y_batch):
        batch_size = x_batch[0].size()[0] # get dynamic batch size
        w_enc_sum = Variable(torch.FloatTensor(batch_size,
            self.step_model.enc0.size()[1]).cuda().zero_())
        conf_sum = Variable(torch.FloatTensor(batch_size,
            self.step_model.enc0.size()[1]).cuda().zero_())
        if self.use_prior:
            enc0_expand = self.step_model.enc0.repeat(batch_size,1)
            conf0_expand = self.step_model.conf0.repeat(batch_size,1)
            w_enc_sum += enc0_expand * conf0_expand
            conf_sum += conf0_expand
        # i = time step
        for i, (x, y) in enumerate(zip(x_batch, y_batch)):
            if self.use_cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            local_enc, conf = self.step_model(x, y)
            w_enc_sum += local_enc * conf
            conf_sum += conf
        return w_enc_sum / conf_sum

""" MODELS """

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

class BasicEncNet(nn.Module):
    def __init__(self, enc_size, x_size, y_size, use_lstm=False):
        super(BasicEncNet, self).__init__()
        self.use_lstm = use_lstm

        enc_weights = [enc_size] * 3
        enc_weights = [x_size+y_size+enc_size] + enc_weights
        # inp_weights = [x_size+y_size] + [enc_size] * 4

        if use_lstm:
            self.lstm1 = nn.LSTMCell(x_size+y_size, enc_size)
            self.lstm2 = nn.LSTMCell(enc_size, enc_size)
        else:
            self.enc_linears = nn.ModuleList([
                nn.Linear(inp, out) for inp, out in zip(enc_weights[:-1], enc_weights[1:])])

    def forward(self, x, y, enc0):
        h = torch.cat((x, y, enc0), dim=1)
        if self.use_lstm: # currently not working
            # h1, c1 = self.lstm1(xy, (h0, c0))
            # h2, c2 = self.lstm2(h1, (h1, c1))
            # return h2, c2
            raise NotImplementedError
        else:
            for i, lin in enumerate(self.enc_linears):
                if i == len(self.enc_linears) - 1: 
                    h = lin(h)
                else:
                    h = F.relu(lin(h))

            return h

class ParallelEncNet(nn.Module):
    def __init__(self, widths, x_size, y_size):
        super(ParallelEncNet, self).__init__()

        self.enc0 = Variable(torch.Tensor(np.zeros((1, widths[-1]))).cuda())
        self.conf0 = Variable(torch.Tensor(np.ones((1, widths[-1]))).cuda())

        enc_weights = [x_size+y_size] + widths

        self.enc_linears = nn.ModuleList([
                nn.Linear(inp, out) for inp, out in zip(enc_weights[:-1], enc_weights[1:])])
        self.conf_linears = nn.ModuleList([
                nn.Linear(inp, out) for inp, out in zip(enc_weights[:-1], enc_weights[1:])])

    def forward(self, x, y):
        h = torch.cat((x, y), dim=1)
        for i, lin in enumerate(self.enc_linears):
            if i == len(self.enc_linears) - 1: 
                h = lin(h)
            else:
                h = F.relu(lin(h))
        conf = torch.cat((x, y), dim=1)
        for i, lin in enumerate(self.conf_linears):
            if i == len(self.conf_linears) - 1: 
                conf = torch.sigmoid(lin(conf))
            else:
                conf = F.relu(lin(conf))

        return h, conf

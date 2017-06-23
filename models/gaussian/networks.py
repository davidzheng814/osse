import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

def get_encoder(model, args, x_size, y_size):
    if model == 'ff':
        return BasicEncNet(args.widths[-1], x_size, y_size, use_lstm=False)
    elif model == 'lstm':
        return BasicEncNet(args.widths[-1], x_size, y_size, use_lstm=True)
    elif model == 'parallel':
        return ParallelEncNet(args.widths, x_size, y_size)
    raise RuntimeError("Encoder " + model + " not found!")

def get_predictor(model, args, x_size, y_size):
    if model == 'basic':
        return BasicPredictNet(args.widths[-1], x_size, y_size)
    raise RuntimeError("Predictor " + model + " not found!")

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

        enc_weights = [x_size+y_size] + widths
        # inp_weights = [x_size+y_size] + [enc_size] * 4

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

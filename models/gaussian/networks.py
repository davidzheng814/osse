import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

def get_encoder(model, args, x_size, y_size):
    if model == 'ff':
        return BasicEncNet(args.enc_size, x_size, y_size, use_lstm=False)
    elif model == 'lstm':
        return BasicEncNet(args.enc_size, x_size, y_size, use_lstm=True)
    elif model == 'parallel':
        return LinComEncNet(args.enc_size, x_size, y_size)
    raise RuntimeError("Encoder " + model + " not found!")

def get_predictor(model, args, x_size, y_size):
    if model == 'basic':
        return BasicPredictNet(args.enc_size, x_size, y_size)
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

class LinComEncNet(nn.Module):
    def __init__(self, enc_size, x_size, y_size):
        super(EncNet, self).__init__()

        enc_weights = [enc_size] * 3
        enc_weights = [x_size+y_size+enc_size] + enc_weights
        # inp_weights = [x_size+y_size] + [enc_size] * 4

        self.lstm1 = nn.LSTMCell(x_size+y_size, enc_size)
        self.lstm2 = nn.LSTMCell(enc_size, enc_size)

    def forward(self, x, y, enc0):
        h = torch.cat((x, y, enc0), dim=1)
        for i, lin in enumerate(self.enc_linears):
            if i == len(self.enc_linears) - 1: 
                h = lin(h)
            else:
                h = F.relu(lin(h))

        return h

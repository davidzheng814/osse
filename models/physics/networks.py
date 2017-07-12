import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

class ParallelEncNet(nn.Module):
    def __init__(self, widths, x_size):
        super(ParallelEncNet, self).__init__()

        enc_weights = [x_size] + widths + [1]

        self.enc_linears = nn.ModuleList([
            nn.Linear(inp, out) for inp, out in zip(enc_weights[:-1], enc_weights[1:])])

        self.conf_linears = nn.ModuleList([
            nn.Linear(inp, out) for inp, out in zip(enc_weights[:-1], enc_weights[1:])])


    def forward(self, x):
        h = x
        for i, lin in enumerate(self.enc_linears):
            if i == len(self.enc_linears) - 1:
                h = lin(h)
            else:
                h = F.relu(lin(h))

        conf = x
        for i, lin in enumerate(self.conf_linears):
            if i == len(self.conf_linears) - 1:
                conf = torch.sigmoid(lin(conf))
            else:
                conf = F.relu(lin(conf))

        return h, conf


'''Gaussian prediction using currently input only.
'''

import argparse
import os
from os.path import join
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

ROOT = '../..'
parser = argparse.ArgumentParser(description='Naive Model for Gaussian Prediction.')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
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
parser.add_argument('--start-train-ind', type=int, default=50,
                    help='index of each group element to start backproping.')
parser.add_argument('--start-test-ind', type=int, default=50,
                    help='index of each group element to start testing.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

class GaussianDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        super(GaussianDataset, self).__init__()
        self.files = glob.glob(args.data_dir + '/*.npz')
        if args.max_files > 0:
            self.files = self.files[:args.max_files]

        assert len(self.files) > args.test_files

        if train:
            self.files = self.files[:-args.test_files]
        else:
            self.files = self.files[-args.test_files:]

        with np.load(self.files[0]) as data:
            self.x_size = int(data['x_size'])
            self.y_size = int(data['y_size'])
            self.data = [GaussianDataset.to_list(data)]

        for file_ in self.files[1:]:
            with np.load(file_) as data:
                assert data['x_size'] == self.x_size and data['y_size'] == self.y_size
                self.data.append(GaussianDataset.to_list(data))

    @staticmethod
    def to_list(data):
        to_torch_tensor = lambda x: torch.from_numpy(x.astype(np.float32))
        return [(to_torch_tensor(x), to_torch_tensor(y)) for x, y in zip(data['x'], data['y'])]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]


class Net(nn.Module):
    def __init__(self, x_size, y_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(x_size+y_size, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, y_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_set = GaussianDataset(train=True)
train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_set = GaussianDataset(train=False)
test_loader = torch.utils.data.DataLoader(test_set,
    batch_size=args.batch_size, shuffle=False, **kwargs)
x_size, y_size = train_set.x_size, train_set.y_size
model = Net(x_size, y_size)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

def train_epoch(epoch):
    tot_loss = 0
    num_samples = 0
    for batch_idx, batch in enumerate(train_loader):
        batch_loss = Variable(torch.FloatTensor(1).zero_())
        optimizer.zero_grad()
        last_y = Variable(batch[args.start_train_ind-1][1])
        for x, y in batch[args.start_train_ind:]:
            x, y = Variable(x), Variable(y)
            pred = model(torch.cat((x, last_y), dim=1))
            loss = criterion(pred, y)
            batch_loss += loss
            tot_loss += loss.data[0]
            num_samples += args.batch_size
            last_y = y
        batch_loss.backward()
        optimizer.step()

    print "Train Loss:", tot_loss / num_samples, 

def test_epoch(epoch):
    tot_loss = 0
    num_samples = 0
    for batch_idx, batch in enumerate(test_loader):
        last_y = Variable(batch[args.start_test_ind-1][1])
        for x, y in batch[args.start_test_ind:]:
            x, y = Variable(x), Variable(y)
            pred = model(torch.cat((x, last_y), dim=1))
            loss = criterion(pred, y)
            tot_loss += loss.data[0]
            num_samples += args.batch_size
            last_y = y

    print "Test Loss:", tot_loss / num_samples

if __name__ == '__main__':
    for epoch in range(1, args.epochs+1):
        train_epoch(epoch)
        test_epoch(epoch)

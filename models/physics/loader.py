import numpy as np
import torch
import h5py

import random
import glob

MAX_MASS = 12.

# Will randomly shuffle cached data
def get_loader_or_cache(loader, cache):
    if cache is None:
        return loader
    elif len(cache) == 0:
        return CachedDataLoader(loader, cache)
    else:
        random.shuffle(cache)
        return cache

class CachedDataLoader:
    def __init__(self, loader, cache):
        self.loader_iter = loader.__iter__()
        self.cache = cache

    def __len__(self):
        return len(self.loader_iter)

    def __iter__(self):
        return self

    def next(self):
        result = self.loader_iter.next()
        self.cache += [result]
        return result

# For full end to end models (model.py)
class PhysicsDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, num_points, num_test_points, train=True):
        super(PhysicsDataset, self).__init__()

        assert (num_points >= num_test_points) or (num_points < 0)

        f = h5py.File(data_file, 'r')

        num_points = num_points if num_points >= 0 else f['enc_x'].shape[0]

        if train:
            self.enc_x = f['enc_x'][:num_points - num_test_points]
            self.pred_xs = f['pred_x'][:num_points - num_test_points]
            self.y = f['y'][:num_points - num_test_points] / MAX_MASS
        else:
            self.enc_x = f['enc_x'][num_points - num_test_points:num_points]
            self.pred_xs = f['pred_x'][num_points - num_test_points:num_points]
            self.y = f['y'][num_points - num_test_points:num_points] / MAX_MASS

            # long_x and long_y are prepared as torch tensors. x and y are just numpy arrays.
            self.enc_x_long = torch.Tensor(f['enc_x_long'][:])
            self.pred_xs_long = torch.Tensor(f['pred_x_long'][:])
            self.y_long = torch.Tensor(f['y_long'][:] / MAX_MASS)

        self.n_objects = self.enc_x.shape[2]
        self.y_size = self.y.shape[1] / self.n_objects
        self.state_size = self.enc_x.shape[3]

    def __len__(self):
        return len(self.enc_x)

    def __getitem__(self, key):
        return self.enc_x[key], self.pred_xs[key], self.y[key]

# For infer_model.py
class EncPhysicsDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, num_points, num_test_points, num_sequential_frames,
            train=True):
        super(EncPhysicsDataset, self).__init__()

        assert (num_points >= num_test_points) or (num_points < 0)

        f = h5py.File(data_file, 'r')

        num_points = num_points if num_points >= 0 else f['x'].shape[0]

        if train:
            self.x = f['x'][:num_points - num_test_points]
            self.y = f['y'][:num_points - num_test_points]
        else:
            self.x = f['x'][num_points - num_test_points:num_points]
            self.y = f['y'][num_points - num_test_points:num_points]

        self.n_objects = self.x.shape[2]
        self.state_size = self.x.shape[3]

        self.num_sequential_frames = num_sequential_frames
        self.x_size = self.x.shape[2] * self.x.shape[3] * num_sequential_frames
        self.y_size = self.y.shape[1] - 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        x = self.x[key].astype(np.float32)
        T, _, _ = x.shape
        x = x.reshape(T, -1)
        x_list = []
        for i in xrange(len(x) - self.num_sequential_frames):
            x_list.append(x[i:i+self.num_sequential_frames,:].flatten())
        y = self.y[key].astype(np.float32)[1:] # Only last two masses

        return x_list, y 


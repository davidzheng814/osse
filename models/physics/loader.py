import numpy as np
import torch
import h5py

import random
import glob

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

class PhysicsDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, num_points, num_test_points, train=True):
        super(PhysicsDataset, self).__init__()

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

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        x = list(self.x[key].astype(np.float32))

        return x


import numpy as np
import torch

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
    def __init__(self, data_dir, num_files, num_test_files, train=True):
        super(PhysicsDataset, self).__init__()
        self.files = glob.glob(data_dir + '/*.npz')

        if num_files > 0:
            self.files = self.files[:num_files]

        assert len(self.files) > num_test_files

        self.train = train
        if train:
            self.files = self.files[:-num_test_files]
        else:
            self.files = self.files[-num_test_files:]

        with np.load(self.files[0]) as data:
            self.n_objects = data['x'].shape[1]
            self.state_size = data['x'].shape[2]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key):
        with np.load(self.files[key]) as data:
            x = list(data['x'].astype(np.float32))

        return x


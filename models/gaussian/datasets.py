'''Collection of datasets and a method to construct them.
'''

import glob
import numpy as np
import math

import torch
import torch.utils.data

class GaussianDataset(torch.utils.data.Dataset):
    def __init__(self, args, train=True, normalize=False):
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
            if not train:
                self.weights = [data['W']]

        for file_ in self.files[1:]:
            with np.load(file_) as data:
                assert data['x_size'] == self.x_size and data['y_size'] == self.y_size
                datadict = {key: value for key, value in data.iteritems()}
                data = datadict
                if normalize:
                    data['y'] = data['y'] / np.max(np.abs(data['y']))
                    data['x'] = data['x'] / np.max(np.abs(data['x']))
                self.data.append(GaussianDataset.to_list(data))
                if not train:
                    self.weights.append(data['W'])

        if not train:
            self.weights = np.array(self.weights)
            num_batches = int(math.ceil(len(self.weights) / float(args.batch_size)))
            self.weights = np.array_split(self.weights, num_batches, axis=0)
            self.weights = [torch.from_numpy(x.astype(np.float32)).cuda() for x in self.weights]

    @staticmethod
    def to_list(data):
        to_torch_tensor = lambda x: torch.from_numpy(x.astype(np.float32))
        return [(to_torch_tensor(x), to_torch_tensor(y)) for x, y in zip(data['x'], data['y'])]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]


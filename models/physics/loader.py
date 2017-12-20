import numpy as np 
import h5py

import random
import glob

# For full end to end models (model.py)
class PhysicsDataset(object):
    def __init__(self, data_file, num_points, num_test_points, batch_size, train=True, maxes=None):
        assert (num_points >= num_test_points) or (num_points < 0)

        f = h5py.File(data_file, 'r')

        num_points = num_points if num_points >= 0 else f['obs_x'].shape[0]

        if train:
            self.obs_x_true = f['obs_x'][:num_points - num_test_points]
            self.ro_x_true = f['ro_x'][:num_points - num_test_points]
            self.y_true = f['y'][:num_points - num_test_points]
        else:
            self.obs_x_true = f['obs_x'][-num_test_points:]
            self.ro_x_true = f['ro_x'][-num_test_points:]
            self.y_true = f['y'][-num_test_points:]

            self.obs_x_true_long = f['obs_x_long'][:]
            self.ro_x_true_long = f['ro_x_long'][:]
            self.y_true_long = f['y_long'][:]
            self.n_ro_frames_long = self.ro_x_true_long.shape[2]

        f.close()

        if maxes:
            self.maxes = maxes
        else:
            self.maxes = {
                'state':np.amax(self.obs_x_true, axis=(0, 1, 2)), 
                'y':np.amax(self.y_true, axis=(0, 1))
            }

        self.obs_x_true /= self.maxes['state']
        self.ro_x_true /= self.maxes['state']
        self.y_true /= self.maxes['y']

        if len(self.y_true.shape) == 3:
            self.y_true = np.stack(
                    [self.y_true[:,0,1], self.y_true[:,0,2], self.y_true[:,1,2]],
                    axis=1)
            if not train:
                self.y_true_long = np.stack(
                    [self.y_true_long[:,0,1], self.y_true_long[:,0,2], self.y_true_long[:,1,2]],
                    axis=1)

        self.batch_size = batch_size
        self.train = train

        self.n_obs_frames = self.obs_x_true.shape[1]
        self.n_objects = self.obs_x_true.shape[2]
        self.state_size = self.obs_x_true.shape[3]

        self.n_rollouts = self.ro_x_true.shape[1]
        self.n_ro_frames = self.ro_x_true.shape[2]
        self.y_size = self.y_true.shape[1] // self.n_objects

    def __len__(self):
        return len(self.obs_x_true)

    def get_batches(self):
        inds = np.arange(len(self.obs_x_true))
        if self.train:
            np.random.shuffle(inds)

        ind = 0
        while ind < len(inds):
            yield (self.obs_x_true[inds[ind:ind+self.batch_size]],
                   self.ro_x_true[inds[ind:ind+self.batch_size]],
                   self.y_true[inds[ind:ind+self.batch_size]])
            ind += self.batch_size

        return

    def get_long_batch(self):
        return self.obs_x_true_long, self.ro_x_true_long, self.y_true_long

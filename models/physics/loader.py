import numpy as np
import h5py

import random
import numpy
import glob

MAX_MASS = 12.

# For full end to end models (model.py)
class PhysicsDataset(object):
    def __init__(self, data_file, num_points, num_test_points, batch_size, train=True):
        assert (num_points >= num_test_points) or (num_points < 0)

        f = h5py.File(data_file, 'r')

        num_points = num_points if num_points >= 0 else f['enc_x'].shape[0]

        if train:
            self.obs_x_true = f['enc_x'][:num_points - num_test_points]
            self.ro_x_true = f['pred_x'][:num_points - num_test_points]
            self.y_true = f['y'][:num_points - num_test_points] / MAX_MASS
        else:
            self.obs_x_true = f['enc_x'][-num_test_points:]
            self.ro_x_true = f['pred_x'][-num_test_points:]
            self.y_true = f['y'][-num_test_points:] / MAX_MASS

            self.obs_x_true_long = f['enc_x_long'][:]
            self.ro_x_true_long = f['pred_x_long'][:]
            self.y_true_long = f['y_long'][:] / MAX_MASS

        f.close()

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



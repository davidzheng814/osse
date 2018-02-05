import numpy as np 
import h5py

import random
import glob

DSET_SFXES = {
    'train':'',
    'test':'',
    'obj3':'_obj3',
    'obj9':'_obj9',
    'long':'_long',
    'mass32':'_mass32'
}

# For full end to end models (model.py)
class PhysicsDataset(object):
    def __init__(self, data_file, dset, batch_size=None, num_points=None, norm_x=None):
        f = h5py.File(data_file, 'r')

        sfx = DSET_SFXES[dset]

        if num_points is not None and dset == 'test':
            self.obs_x = f['obs_x'+sfx][-num_points:]
            self.ro_x = f['ro_x'+sfx][-num_points:]
            self.y = f['y'+sfx][-num_points:]
        elif num_points is not None:
            self.obs_x = f['obs_x'+sfx][:num_points]
            self.ro_x = f['ro_x'+sfx][:num_points]
            self.y = f['y'+sfx][:num_points]
        else:
            self.obs_x = f['obs_x'+sfx][:]
            self.ro_x = f['ro_x'+sfx][:]
            self.y = f['y'+sfx][:]

        f.close()

        if isinstance(norm_x, str) and norm_x == 'use_data':
            norm_x = np.amax(self.obs_x, axis=(0, 1, 2))

        if norm_x is not None:
            self.obs_x /= norm_x
            self.ro_x /= norm_x
            self.norm_x = norm_x

        self.batch_size = batch_size
        self.shuffle = dset == 'train'
        self.dset = dset
        self.num_points = self.obs_x.shape[0]
        self.n_obs_frames = self.obs_x.shape[1]
        self.n_ro_frames = self.ro_x.shape[1]
        self.n_objects = self.obs_x.shape[2]
        self.state_size = self.obs_x.shape[3]

        # TODO Hacky. 
        self.y = self.y[:,:self.n_objects]

        self.y_size = self.y.shape[1] // self.n_objects

        self.y = np.log(self.y)
        self.y -= self.y[:,0:1]

    def __len__(self):
        return self.num_points

    def get_all(self):
        return self.obs_x, self.ro_x, self.y

    def get_batches(self):
        if self.batch_size is None:
            yield self.get_all() # Just a length-1 iterator
            return

        inds = np.arange(self.num_points)
        if self.shuffle:
            np.random.shuffle(inds)

        ind = 0
        while ind < self.num_points:
            yield (self.obs_x[inds[ind:ind+self.batch_size]],
                   self.ro_x[inds[ind:ind+self.batch_size]],
                   self.y[inds[ind:ind+self.batch_size]])
            ind += self.batch_size

        return


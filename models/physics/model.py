from __future__ import print_function, division

import time
import math
import json
import os
import sys
from os.path import basename, join
import shutil

import tensorflow as tf
import numpy as np
from sklearn.linear_model import LinearRegression

from gru_enc_model import gru_enc_net
from inet_enc_model import inet_enc_net
from pred_model import predict_net
from parser import parser
from loader import PhysicsDataset
from shared import log, get_enc_analysis

OUT_WIDTH = 4

class Model(object):
    def get_enc_pred(self, obs_x_true, y_true):
        """Returns enc_pred
            obs_x_true - Shape: [batch_size, n_obs_frames, n_objects, state_size]
            y_true - Shape: [batch_size, n_objects * y_size]
        """
        args = self.args
        n_objects = tf.shape(obs_x_true)[2]

        if args.pred_only:
            enc_pred = tf.log(tf.reshape(y_true, [-1, n_objects, 1])) # TODO Hardcoded y_size.
        else:
            with tf.variable_scope("enc_net", reuse=tf.AUTO_REUSE):
                # Shape: [batch_size, n_objects, enc_size]
                # enc_pred = gru_enc_net(obs_x_true, args.enc_lstm_widths, args.enc_dense_widths)
                assert len(args.enc_dense_widths) == 1
                enc_pred, enc_reg = inet_enc_net(obs_x_true, args.re_widths, args.sd_widths,
                                        args.agg_widths, args.effect_width, args.enc_dense_widths[-1], args.inet_pred_frames)

                if not args.no_ref_enc_sub:
                    enc_pred -= tf.tile(enc_pred[:,:1], [1, n_objects, 1])

                if args.enc_only:
                    return enc_pred, enc_reg

        return enc_pred

    def get_ro_pred(self, ro_x_true, enc_pred):
        """Returns ro_x_pred
        
            ro_x_true - Shape: [batch_size, n_ro_frames, n_objects, state_size]
            enc_pred - Shape: [batch_size, n_objects, enc_size]
        """

        args = self.args
        n_ro_frames = int(ro_x_true.get_shape()[1])
        n_objects = int(ro_x_true.get_shape()[2])
        state_size = int(ro_x_true.get_shape()[3])
        enc_size = int(enc_pred.get_shape()[2])

        if args.baseline:
            # predict the output assuming constant velocity. 
            first = ro_x_true[:,:-1] * self.dset.norm_x # first few frames
            vel = first[:,:,:,2:] 
            pad_vel = tf.concat([vel, tf.zeros(tf.shape(vel))], axis=3)
            ro_x_pred = first + pad_vel
            ro_x_pred /= self.dset.norm_x
        else:
            with tf.variable_scope("predict_net", reuse=tf.AUTO_REUSE):
                ro_x_inp = ro_x_true[:,0]
                ro_x_pred, reg_loss = predict_net(ro_x_inp, enc_pred, n_ro_frames, args.re_widths, args.sd_widths,
                                        args.agg_widths, args.effect_width, OUT_WIDTH,
                                        noise_ratio=args.noise)

        return ro_x_pred, reg_loss

    def get_enc_loss(self, enc_pred, y_true):
        with tf.variable_scope("enc_losses"):
            enc_pred = tf.reshape(enc_pred, [-1, tf.shape(y_true)[1]])
            loss = tf.reduce_mean(tf.squared_difference(enc_pred, y_true))
            self.summaries.append(tf.summary.scalar('enc_loss', loss))

        return loss

    def get_pred_loss(self, ro_x_pred, ro_x_true):
        with tf.variable_scope("losses"):
            # MSE loss of ro_x_true_vel and ro_x_pred
            loss = tf.reduce_mean(tf.squared_difference(
                ro_x_true[:,1:],
                ro_x_pred))
            # TODO Pos loss from before. 

            self.summaries.append(tf.summary.scalar('pred_loss', loss))

        return loss

    def get_optim(self, lr):
        with tf.variable_scope("optim"):
            optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        return optim

    def build_model(self, train=False):
        self.summaries = []
        self.obs_x_true = tf.placeholder(tf.float32,
                [None, self.dset.n_obs_frames, self.dset.n_objects, self.dset.state_size],
                name="obs_x_true")
        self.ro_x_true = tf.placeholder(tf.float32,
                [None, self.dset.n_ro_frames, self.dset.n_objects, self.dset.state_size],
                name="ro_x_true")
        self.y_true = tf.placeholder(tf.float32, [None, self.dset.n_objects], name="y_true")
        self.lr = tf.placeholder(tf.float32, [], name="lr")

        out = self.get_enc_pred(self.obs_x_true, self.y_true)

        if self.args.enc_only:
            self.enc_pred, self.enc_reg_loss = out
            self.pred_loss = self.get_enc_loss(self.enc_pred, self.y_true)
            self.loss = self.pred_loss + self.args.enc_reg_factor * self.enc_reg_loss
        else:
            self.enc_pred = out
            self.ro_x_pred, self.reg_loss = self.get_ro_pred(self.ro_x_true, self.enc_pred)
            self.pred_loss = self.get_pred_loss(self.ro_x_pred, self.ro_x_true)
            self.loss = self.args.reg_factor * self.reg_loss + self.pred_loss
            self.summaries.append(tf.summary.scalar('reg_loss', self.reg_loss))
            self.summaries.append(tf.summary.scalar('loss', self.loss))

        if not self.args.baseline and train:
            self.optim = self.get_optim(self.lr)

        self.summary = tf.summary.merge(self.summaries)

    def save_encs(self, enc_pred, y_true):
        log(self.args.log_dir, "Enc R^2:", get_enc_analysis(enc_pred, y_true))
        np.savez(join(self.args.log_dir, 'enc_'+self.dset_name+'.npz'), enc=enc_pred, y=y_true)

    def save_ro(self):
        pass
        # TODO It's not this easy....
        # for i in range(len(ro_x_true)):
        #     save_json(ro_x_true[i], ro_x_pred[i], y_true[i],
        #               join(self.args.log_dir, 'ro_'+str(i)+'.json'))

class TrainModel(Model):
    def __init__(self, args):
        self.args = args
        self.lr_val = self.args.lr

        assert args.num_points == 0 or args.num_points > args.test_points
        self.dset = PhysicsDataset(args.data_file, 'train',
            batch_size=args.batch_size,
            num_points=args.num_points - args.test_points,
            norm_x='use_data')
        if args.enc_only:
            assert args.enc_dense_widths[-1] == self.dset.y_size
        self.batches_per_epoch = len(self.dset) // args.batch_size
        self.build_model(train=True)

    def set_learning_rate(self, epochs_without_dec):
        if self.args.decay_cutoff > 0 and epochs_without_dec >= self.args.decay_cutoff:
            self.lr_val *= self.args.decay_factor
            log(self.args.log_dir, 'Decaying learning rate to {:.2E}'.format(self.lr_val))

    def run(self, train_ind, sess, writer, epochs_without_dec):
        self.set_learning_rate(epochs_without_dec)
        reg_losses, pred_losses = [], []
        for ind, (obs_x_true, ro_x_true, y_true) in enumerate(self.dset.get_batches()):
            if not self.args.enc_only:
                reg_loss, pred_loss, summary, _ = sess.run([self.reg_loss, self.pred_loss, self.summary, self.optim], feed_dict={
                    self.obs_x_true:obs_x_true,
                    self.ro_x_true:ro_x_true,
                    self.y_true:y_true,
                    self.lr:self.lr_val
                })
            else:
                # TODO: This isn't really pred loss but enc loss, but report for simplicity
                reg_loss, pred_loss, summary, _ = sess.run([self.enc_reg_loss, self.pred_loss, self.summary, self.optim],
                    feed_dict={
                        self.obs_x_true:obs_x_true,
                        self.ro_x_true:ro_x_true,
                        self.y_true:y_true,
                        self.lr:self.lr_val
                    })

            reg_losses.append(reg_loss)
            pred_losses.append(pred_loss)
            writer.add_summary(summary, train_ind + ind)

        log(self.args.log_dir, 'Train - Reg Loss: {:.4E} Pred Loss: {:.4E}'.format(np.mean(reg_losses), np.mean(pred_losses)))

class TestModel(Model):
    def __init__(self, args, dset_name, norm_x):
        self.args = args
        self.dset = PhysicsDataset(args.data_file, dset_name,
            batch_size=args.batch_size,
            num_points=args.test_points,
            norm_x=norm_x)
        self.best_loss = float('inf')
        self.dset_name = dset_name
        self.build_model(train=False)
        self.epochs_wo_dec = 0

    def run(self, train_ind, sess, writer, save_encs=False, save_ro=False):
        if save_ro:
            self.save_ro()
            return

        reg_losses, pred_losses, enc_preds, y_trues = [], [], [], []
        for obs_x_true, ro_x_true, y_true in self.dset.get_batches():
            if not self.args.enc_only:
                reg_loss, pred_loss, enc_pred, summary = sess.run([self.reg_loss, self.pred_loss, self.enc_pred, self.summary], feed_dict={
                    self.obs_x_true:obs_x_true,
                    self.ro_x_true:ro_x_true,
                    self.y_true:y_true,
                })
            else:
                # TODO: This isn't really pred loss but enc loss, but report for simplicity
                reg_loss, pred_loss, enc_pred, summary = sess.run([self.enc_reg_loss, self.pred_loss, self.enc_pred, self.summary],
                    feed_dict={
                        self.obs_x_true:obs_x_true,
                        self.ro_x_true:ro_x_true,
                        self.y_true:y_true,
                    })

            reg_losses.append(reg_loss)
            pred_losses.append(pred_loss)
            enc_preds.append(enc_pred)
            y_trues.append(y_true)

            if self.dset_name == 'test': # TODO add summary for all dsets, not just test
                writer.add_summary(summary, train_ind)

        reg_loss = np.mean(reg_losses)
        pred_loss = np.mean(pred_losses)
        enc_pred = np.concatenate(enc_preds)
        y_true = np.concatenate(y_trues)

        log(self.args.log_dir, 'Dset: {} Reg Loss: {:.4E} Pred Loss: {:.4E}'.format(self.dset_name, reg_loss, pred_loss))

        if save_encs: # Saves and analyzes encodings
            self.save_encs(enc_pred, y_true)

        # Update epochs_wo_dec
        if pred_loss < self.best_loss:
            self.epochs_wo_dec = 0
            self.best_loss = pred_loss
        else:
            self.epochs_wo_dec += 1


from __future__ import print_function, division

print("Importing")

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

from model import TrainModel, TestModel
from parser import parser
from shared import log

args = parser.parse_args()

""" SETUP LOG DIR AND CHECKPOINT DIR """

if args.restore:
    if args.new_dir:
        old_folder_ind = basename(os.path.dirname(args.restore))
        folder_ind = str(max([int(x) for x in os.listdir(args.log_dir)]) + 1)
        print("Copying {} to {}.".format(old_folder_ind, folder_ind))
        try:
            shutil.copytree(join(args.log_dir, old_folder_ind),
                            join(args.log_dir, folder_ind))
        except shutil.Error as e:
            print("Shutil error but continuing: {}".format(e))
        try:
            shutil.copytree(join(args.ckpt_dir, old_folder_ind),
                            join(args.ckpt_dir, folder_ind))
        except shutil.Error as e:
            print("Shutil error but continuing: {}".format(e))
    else:
        folder_ind = basename(os.path.dirname(args.restore))
else:
    folder_ind = str(max([int(x) for x in os.listdir(args.log_dir)]) + 1)

args.log_dir = join(args.log_dir, folder_ind)
args.ckpt_dir = join(args.ckpt_dir, folder_ind)

if args.restore:
    log(args.log_dir, "Reloading:", folder_ind)
else:
    os.mkdir(args.log_dir)

""" BUILD MODELS """

def build_models():
    print("Loading datasets and building model")

    train_model = TrainModel(args)
    test_model = TestModel(args, 'test', train_model.dset.norm_x)
    obj3_model = TestModel(args, 'obj3', train_model.dset.norm_x)
    obj9_model = TestModel(args, 'obj9', train_model.dset.norm_x)
    long_model = TestModel(args, 'long', train_model.dset.norm_x)
    mass32_model = TestModel(args, 'mass32', train_model.dset.norm_x)

    return train_model, test_model, obj3_model, obj9_model, long_model, mass32_model

""" RUN """

def run(train_model, test_model, obj3_model, obj9_model, long_model, mass32_model):
    with tf.Session() as sess:
        """ Initialize summary op and saver """
        if not args.baseline:
            saver = tf.train.Saver(var_list=tf.trainable_variables())

        """ Initialize variables and start epoch """
        sess.run(tf.global_variables_initializer())
        if args.restore:
            saver.restore(sess, args.restore)
            start_epoch = int(basename(args.restore.split('-')[-1])) + 1
            log(args.log_dir, "Start Epoch: ", start_epoch)
        else:
            start_epoch = 1

        """ Setup tensorboard writers """
        train_writer = tf.summary.FileWriter(args.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(args.log_dir + '/test')

        """ Run epochs """
        for epoch in range(start_epoch, args.epochs+1):
            log(args.log_dir, 'Epoch: {} Log: {}'.format(epoch, folder_ind))
            start_time = time.time()

            train_ind = (epoch-1) * train_model.batches_per_epoch # Set train_ind for train summaries.
            if not args.skip_train and not args.baseline:
                train_model.run(train_ind, sess, train_writer, test_model.epochs_wo_dec)
            train_ind = epoch * train_model.batches_per_epoch # Set train_ind for test summaries.
            test_model.run(train_ind, sess, test_writer, save_encs=True)
            #TODO None of these currently log to tensorboard
            #TODO Fix logging to look nicer, especially obj9 R^2
            obj3_model.run(train_ind, sess, test_writer, save_encs=True)
            obj9_model.run(train_ind, sess, test_writer, save_encs=True)
            mass32_model.run(train_ind, sess, test_writer, save_encs=True)
            long_model.run(train_ind, sess, test_writer, save_ro=True)

            epoch_time = time.time() - start_time
            log(args.log_dir, 'Time: {:.2f}s Epochs w/o Dec: {}\n'.format(epoch_time, test_model.epochs_wo_dec))

            if args.baseline: # baseline only runs for one epoch
                break

            if test_model.epochs_wo_dec == 0: # Only save if model has the best test loss
                saver.save(sess, join(args.ckpt_dir, 'model'), epoch)

if __name__ == '__main__':
    log(args.log_dir, 'python', ' '.join(sys.argv))
    log(args.log_dir, args)
    models = build_models()
    run(*models)


from __future__ import print_function, division

print("Importing")

import time
import math
import os
from os.path import basename

import tensorflow as tf
import numpy as np

from enc_model import lstm_enc_net
from pred_model import predict_net
from parser import parser
from loader import PhysicsDataset

args = parser.parse_args()

print("Loading Datasets")

train_set = PhysicsDataset(args.data_file, args.num_points, args.test_points, args.batch_size, train=True)
test_set = PhysicsDataset(args.data_file, args.num_points, args.test_points, args.batch_size, train=False)

n_obs_frames = train_set.n_obs_frames
n_ro_frames = train_set.n_ro_frames
n_objects = train_set.n_objects
n_rollouts = train_set.n_rollouts
state_size = train_set.state_size
y_size = train_set.y_size
enc_size = args.enc_dense_widths[-1] // n_objects

def get_model_pred(obs_x_true, ro_x_true):
    with tf.variable_scope("enc_net"):
        enc_pred = lstm_enc_net(obs_x_true, args.enc_lstm_widths, args.enc_dense_widths)
        enc_pred = tf.tile(tf.expand_dims(enc_pred, axis=1), [1, n_rollouts, 1, 1])
        enc_pred = tf.reshape(enc_pred, [-1, n_objects, enc_size])

    with tf.variable_scope("preprocess_prednet"):
        n_prep_frames = max(args.offsets) + args.frames_per_samp - 1
        init_ro_state = ro_x_true[:,:,:n_prep_frames]
        init_ro_state = tf.reshape(init_ro_state, [-1, n_prep_frames, n_objects, state_size])

    with tf.variable_scope("predict_net"):
        ro_x_pred = predict_net(init_ro_state, enc_pred,
                args.frames_per_samp, args.code_size, n_ro_frames, args.offsets)

    return enc_pred, ro_x_pred

def get_loss_and_optim(ro_x_pred, ro_x_true, discount):
    with tf.variable_scope("losses"):
        discount = tf.reshape(discount, [-1, 1, 1])
        ro_x_true = tf.reshape(ro_x_true, [-1, n_ro_frames, n_objects, state_size])
        loss_tensor = tf.squared_difference(ro_x_true, ro_x_pred)
        loss = tf.reduce_mean(discount * loss_tensor, name="loss")
        tf.summary.scalar('loss', loss)
        pos_loss = tf.reduce_mean(loss_tensor[:,:,:,:2], name="pos_loss")
        tf.summary.scalar('pos_loss', pos_loss)

    with tf.variable_scope("optim"):
        optim = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss)
    
    return loss, pos_loss, optim

def get_discount_factor():
    if args.beta:
        epochs = (train_iter+1) * args.batch_size / len(train_set)
        return 1 - math.e ** (-epochs / args.beta)
    return 1.

print("Building Model")

obs_x_true = tf.placeholder(tf.float32, [None, n_obs_frames, n_objects, state_size],
        name="obs_x_true")
ro_x_true = tf.placeholder(tf.float32, [None, n_rollouts, n_ro_frames, n_objects, state_size],
        name="ro_x_true")
y_true = tf.placeholder(tf.float32, [None, n_objects], name="y_true")
discount = tf.placeholder(tf.float32, [n_ro_frames], name="discount")

enc_pred, ro_x_pred = get_model_pred(obs_x_true, ro_x_true)
loss, pos_loss, optim = get_loss_and_optim(ro_x_pred, ro_x_true, discount)

train_iter = 0
summary = tf.summary.merge_all()

new_folder = str(max([int(x) for x in os.listdir("logs/")]) + 1)
args.log_dir = "logs/" + new_folder

def run_epoch(sess, writer, train):
    global train_iter
    epoch_loss, epoch_pos_loss, num_batches = 0., 0., 0
    data_set = train_set if train else test_set
    out_ops = [loss, pos_loss, summary]
    if train:
        out_ops.append(optim)

    for obs_x_true_, ro_x_true_, y_true_ in data_set.get_batches():
        discount_ = np.geomspace(1, get_discount_factor() ** n_ro_frames, num=n_ro_frames)

        kwargs = {}
        if args.runtime and train and train_iter % 100 == 0: # record runtime
            run_metadata = tf.RunMetadata()
            kwargs = {
                'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                'run_metadata': run_metadata
            }

        res = sess.run(out_ops, feed_dict={
            obs_x_true:obs_x_true_,
            ro_x_true:ro_x_true_,
            discount:discount_
        }, **kwargs)

        if args.runtime and train and train_iter % 100 == 0:
            writer.add_run_metadata(run_metadata, 'step%d' % train_iter)

        num_batches += 1
        epoch_loss += res[0]
        epoch_pos_loss += res[1]
        writer.add_summary(res[2], train_iter)

        if train:
            train_iter += 1

    return epoch_loss / num_batches, epoch_pos_loss / num_batches

def run():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(args.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(args.log_dir + '/test')

        print("Training")

        for epoch in range(1, args.epochs+1):
            start_time = time.time()
            loss_, pos_loss_ = run_epoch(sess, train_writer, train=True)
            print('Log: {} Discount Factor: {}'.format(
                basename(args.log_dir), get_discount_factor()))
            print('TRAIN: Epoch: {} Total Loss: {:.5f} Pos Loss: {:.5f}  Time: {:.2f}s'.format(
                epoch, loss_, pos_loss_, time.time() - start_time))

            loss_, pos_loss_ = run_epoch(sess, test_writer, train=False)
            print('TEST: Total Loss: {:.5f} Pos Loss: {:.5f}\n'.format(loss_, pos_loss_))

if __name__ == '__main__':
    run()


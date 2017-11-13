from __future__ import print_function, division

print("Importing")

import time
import math
import os
from os.path import basename, join

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
n_prep_frames = max(args.offsets) + args.frames_per_samp - 1

def get_model_pred(obs_x_true, ro_x_true):
    if args.baseline:
        ro_x_pred = tf.reshape(ro_x_true, [-1, n_ro_frames, n_objects, state_size])
        ro_x_pred = tf.concat(
                [ro_x_pred[:,:n_prep_frames], ro_x_pred[:,n_prep_frames-1:-1]], axis=1)

        return None, ro_x_pred

    with tf.variable_scope("enc_net"):
        enc_pred = lstm_enc_net(obs_x_true, args.enc_lstm_widths, args.enc_dense_widths)
        enc_pred_expand = tf.tile(tf.expand_dims(enc_pred, axis=1), [1, n_rollouts, 1, 1])
        enc_pred_expand = tf.reshape(enc_pred_expand, [-1, n_objects, enc_size])

    with tf.variable_scope("preprocess_prednet"):
        init_ro_state = ro_x_true[:,:,:n_prep_frames]
        init_ro_state = tf.reshape(init_ro_state, [-1, n_prep_frames, n_objects, state_size])

    with tf.variable_scope("predict_net"):
        ro_x_pred, ro_aux_loss = predict_net(init_ro_state, enc_pred_expand,
                args.frames_per_samp, args.code_size, n_ro_frames, args.offsets)

    return enc_pred, ro_x_pred, ro_aux_loss

def get_loss_and_optim(ro_x_pred, ro_x_true, ro_aux_loss, discount):
    with tf.variable_scope("losses"):
        discount = tf.reshape(discount, [-1, 1, 1]) # For broadcasting purposes. 
        ro_x_true = tf.reshape(ro_x_true, [-1, n_ro_frames, n_objects, state_size])
        loss_tensor = tf.squared_difference(ro_x_true, ro_x_pred)

        loss = tf.reduce_mean(discount * loss_tensor, name="loss") + ro_aux_loss
        pos_loss = tf.reduce_mean(loss_tensor[:,n_prep_frames:,:,:2], name="pos_loss")

        tf.summary.scalar('aux_loss', ro_aux_loss)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('pos_loss', pos_loss)

    if args.baseline:
        return loss, pos_loss, None

    with tf.variable_scope("optim"):
        optim = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss)
    
    return loss, pos_loss, optim

def get_enc_corr(enc_pred, y_true):
    mean, var = tf.nn.moments(enc_pred, axes=[0], keep_dims=True)
    enc_pred_std = (enc_pred - mean) / tf.sqrt(var)
    mean, var = tf.nn.moments(y_true, axes=[0], keep_dims=True)
    y_true_std = tf.expand_dims((y_true - mean) / tf.sqrt(var), axis=2)

    enc_corr = tf.reduce_max(tf.abs(tf.reduce_mean(enc_pred_std * y_true_std, axis=0)), axis=1)

    return enc_corr

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

enc_pred, ro_x_pred, ro_aux_loss = get_model_pred(obs_x_true, ro_x_true)
loss, pos_loss, optim = get_loss_and_optim(ro_x_pred, ro_x_true, ro_aux_loss, discount)
enc_corr = get_enc_corr(enc_pred, y_true)

train_iter = 0
summary = tf.summary.merge_all()

new_folder = str(max([int(x) for x in os.listdir(args.log_dir)]) + 1)
args.log_dir = join(args.log_dir, new_folder)

def run_epoch(sess, writer, train):
    global train_iter
    epoch_loss, epoch_pos_loss, epoch_aux_loss, epoch_corr, num_batches = 0., 0., 0., 0., 0
    data_set = train_set if train else test_set
    out_ops = [loss, pos_loss, ro_aux_loss, summary]
    if train and not args.baseline:
        out_ops.append(optim)
    else:
        out_ops.append(enc_corr)

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
            discount:discount_,
            y_true:y_true_,
        }, **kwargs)

        if args.runtime and train and train_iter % 100 == 0:
            writer.add_run_metadata(run_metadata, 'step%d' % train_iter)

        num_batches += 1
        epoch_loss += res[0]
        epoch_pos_loss += res[1]
        epoch_aux_loss += res[2]
        writer.add_summary(res[3], train_iter)

        if train:
            train_iter += 1
        else:
            epoch_corr += res[4]

    if not train:
        print("Enc Corr:", epoch_corr / num_batches)

    return epoch_loss / num_batches, epoch_pos_loss / num_batches, epoch_aux_loss / num_batches

def run():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(args.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(args.log_dir + '/test')

        print("Training")

        for epoch in range(1, args.epochs+1):
            start_time = time.time()
            loss_, pos_loss_, aux_loss_ = run_epoch(sess, train_writer, train=True)
            print('Log: {} Discount Factor: {}'.format(
                basename(args.log_dir), get_discount_factor()))
            print('TRAIN: Epoch: {} Total Loss: {:.6E} Pos Loss: {:.6E} Aux Loss: {:.6E} Time: {:.2f}s'.format(
                epoch, loss_, pos_loss_, aux_loss_, time.time() - start_time))

            loss_, pos_loss_, aux_loss_ = run_epoch(sess, test_writer, train=False)
            print('TEST: Total Loss: {:.6E} Pos Loss: {:.6E} Aux Loss: {:.6E} \n'.format(loss_, pos_loss_, aux_loss_))

if __name__ == '__main__':
    run()


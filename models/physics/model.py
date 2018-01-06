from __future__ import print_function, division

print("Importing")

import time
import math
import json
import os
from os.path import basename, join
import shutil

import tensorflow as tf
import numpy as np
from sklearn.linear_model import LinearRegression

from enc_model import lstm_enc_net
from pred_model import predict_net
from parser import parser
from loader import PhysicsDataset

args = parser.parse_args()

print("Loading Datasets")

train_set = PhysicsDataset(args.data_file, args.num_points, args.test_points, args.batch_size, train=True)
test_set = PhysicsDataset(args.data_file, args.num_points, args.test_points, args.batch_size, train=False, maxes=train_set.maxes)

n_obs_frames = train_set.n_obs_frames
n_ro_frames = train_set.n_ro_frames
n_objects = train_set.n_objects
n_rollouts = train_set.n_rollouts
n_ro_frames_long = test_set.n_ro_frames_long
state_size = train_set.state_size
y_size = train_set.y_size
enc_size = args.enc_dense_widths[-1] // n_objects
n_prep_frames = max(args.offsets) + args.frames_per_samp - 1

def log(*text):
    print(*text)
    with open(join(args.log_dir, 'log.txt'), 'a') as f:
        f.write(' '.join([str(x) for x in text]) + '\n')

def get_model_pred(obs_x_true, ro_x_true, n_ro_frames, reuse=False):
    if args.baseline:
        ro_x_pred = tf.reshape(ro_x_true, [-1, n_ro_frames, n_objects, state_size])
        ro_x_pred = tf.concat(
                [ro_x_pred[:,:n_prep_frames], ro_x_pred[:,n_prep_frames-1:-1]], axis=1)

        return None, ro_x_pred, tf.constant(0.)

    with tf.variable_scope("enc_net", reuse=reuse):
        enc_pred = lstm_enc_net(obs_x_true, args.enc_lstm_widths, args.enc_dense_widths)
        enc_pred_expand = tf.tile(tf.expand_dims(enc_pred, axis=1), [1, n_rollouts, 1, 1])
        enc_pred_expand = tf.reshape(enc_pred_expand, [-1, n_objects, enc_size])

    with tf.variable_scope("preprocess_prednet", reuse=reuse):
        init_ro_state = ro_x_true[:,:,:n_prep_frames]
        init_ro_state = tf.reshape(init_ro_state, [-1, n_prep_frames, n_objects, state_size])

    with tf.variable_scope("predict_net", reuse=reuse):
        ro_x_pred, ro_aux_loss = predict_net(init_ro_state, enc_pred_expand,
                args.frames_per_samp, args.code_size, n_ro_frames, args.offsets,
                args.noise)

    return enc_pred, ro_x_pred, ro_aux_loss

def get_loss_and_optim(ro_x_pred, ro_x_true, ro_aux_loss, discount, lr_enc, lr_pred):
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
        enc_optim = (tf.train.AdamOptimizer(learning_rate=lr_enc)
            .minimize(loss, var_list=[v for v in tf.trainable_variables()
                                      if 'enc_net' in v.name]))
        pred_optim = (tf.train.AdamOptimizer(learning_rate=lr_pred)
            .minimize(loss, var_list=[v for v in tf.trainable_variables()
                                      if 'predict_net' in v.name]))

        optim = tf.group(enc_optim, pred_optim, name='optim')
    
    return loss, pos_loss, optim

def enc_analysis(enc_pred, y_true):
    """
        @param enc_pred: [batch_size, n_objects, enc_size]
        @param y_true: [batch_size, n_objects]
    """

    r2s = []
    for obj_ind in range(enc_pred.shape[1]):
        X = enc_pred[:,obj_ind]
        y = y_true[:,obj_ind]

        if args.logy:
            y = np.log(y)

        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        r2s.append(r2)

    # mean = np.mean(enc_pred, axis=0, keepdims=True)
    # std = np.std(enc_pred, axis=0, keepdims=True)
    # enc_pred_std = (enc_pred - mean) / std

    # logy = np.log(y_true)
    # mean = np.mean(logy, axis=0, keepdims=True)
    # std = np.std(logy, axis=0, keepdims=True)
    # y_true_std = np.expand_dims((logy - mean) / std, axis=2)

    # enc_corr = np.max(np.abs(np.mean(enc_pred_std * y_true_std, axis=0)), axis=1)

    return r2s

def get_states_json(states):
    payload = []
    for frame in states:
        pos, vel = [], []
        for obj in frame:
            obj = obj.tolist()
            pos.append({'x':obj[0], 'y':obj[1]})
            vel.append({'x':obj[2], 'y':obj[3]})
        payload.append({'pos':pos, 'vel':vel})
    return payload 

def save_json(x_true, x_pred, y_true, out_file):
    """Write states to json file.
        @param x_true: [n_ro_frames_long, n_objects, state_size]
        @param x_pred: [n_ro_frames_long, n_objects, state_size]
        @param y_true: [n_objects]
    """
    payload = {
        'ro_states': [get_states_json(x_pred)],
        'true_states': get_states_json(x_true),
        'enc_true': y_true.tolist()
    }
    with open(out_file, 'w') as f:
        f.write(json.dumps(payload, indent=4, separators=(',',': ')))

def get_discount_factor():
    if args.beta:
        epochs = (train_iter+1) * args.batch_size / len(train_set)
        return 1 - math.e ** (-epochs / args.beta)
    return 1.

lr_params = { 'decay': 1.0 }

def get_learning_rates():
    if args.freeze_encs:
        return 0, args.lr_pred * lr_params['decay']
    return args.lr_enc * lr_params['decay'], args.lr_pred * lr_params['decay']

print("Building Model")

obs_x_true = tf.placeholder(tf.float32, [None, n_obs_frames, n_objects, state_size],
        name="obs_x_true")
ro_x_true = tf.placeholder(tf.float32, [None, n_rollouts, n_ro_frames, n_objects, state_size],
        name="ro_x_true")
y_true = tf.placeholder(tf.float32, [None, n_objects], name="y_true")
discount = tf.placeholder(tf.float32, [n_ro_frames], name="discount")
lr_enc = tf.placeholder(tf.float32, [], name="lr_enc")
lr_pred = tf.placeholder(tf.float32, [], name="lr_pred")

enc_pred, ro_x_pred, ro_aux_loss = get_model_pred(obs_x_true, ro_x_true, n_ro_frames)
loss, pos_loss, optim = get_loss_and_optim(ro_x_pred, ro_x_true, ro_aux_loss, discount, lr_enc, lr_pred)

if args.long:
    ro_x_true_long = tf.placeholder(tf.float32, 
        [None, n_rollouts, n_ro_frames_long, n_objects, state_size],
        name="ro_x_true_long")
    enc_pred_long, ro_x_pred_long, _ = get_model_pred(obs_x_true, ro_x_true_long, n_ro_frames_long, reuse=True)

train_iter = 0
summary = tf.summary.merge_all()

if not args.baseline:
    saver = tf.train.Saver(var_list=tf.trainable_variables())

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
    log("Reloading:", folder_ind)
else:
    os.mkdir(args.log_dir)

def run_epoch(sess, writer, train):
    global train_iter
    epoch_loss, epoch_pos_loss, epoch_aux_loss, num_batches = 0., 0., 0., 0
    data_set = train_set if train else test_set
    out_ops = [loss, pos_loss, ro_aux_loss, summary]
    if train and not args.baseline:
        out_ops.append(optim)
    else:
        if args.calc_encs:
            out_ops.append(enc_pred)
            y_trues, enc_preds = [], []

    for obs_x_true_, ro_x_true_, y_true_ in data_set.get_batches():
        discount_ = np.geomspace(1, get_discount_factor() ** n_ro_frames, num=n_ro_frames)
        lr_enc_, lr_pred_ = get_learning_rates()

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
            lr_enc:lr_enc_,
            lr_pred:lr_pred_,
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
        elif args.calc_encs:
            y_trues.append(y_true_)
            enc_preds.append(res[4])

    if not train and args.calc_encs: # Encoding analysis
        epoch_y_true = np.concatenate(y_trues, axis=0)
        epoch_enc_pred = np.concatenate(enc_preds, axis=0)
        log("Enc R^2:", enc_analysis(epoch_enc_pred, epoch_y_true))
        np.savez(join(args.log_dir, 'enc.npz'), enc=epoch_enc_pred, y=epoch_y_true)

    return epoch_loss / num_batches, epoch_pos_loss / num_batches, epoch_aux_loss / num_batches

def run_long_rollouts(sess):
    obs_x_true_long_, ro_x_true_long_, y_true_long_ = test_set.get_long_batch()
    ro_x_pred_long_ = sess.run(ro_x_pred_long, feed_dict={
        obs_x_true:obs_x_true_long_,
        ro_x_true_long:ro_x_true_long_,
    })
    ro_x_true_long_ = np.reshape(ro_x_true_long_,
            [-1, n_ro_frames_long, n_objects, state_size])
    y_true_long_ = np.reshape(np.tile(y_true_long_, [1, n_rollouts]),
            [-1, n_objects])

    ro_x_true_long_ *= train_set.maxes['state']
    ro_x_pred_long_ *= train_set.maxes['state']
    y_true_long_ *= train_set.maxes['y']
    for samp_ind in range(len(ro_x_true_long_)):
        save_json(ro_x_true_long_[samp_ind], ro_x_pred_long_[samp_ind],
                y_true_long_[samp_ind],
                join(args.log_dir, 'long_ro_'+str(samp_ind)+'.json'))

def run():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if args.restore:
            saver.restore(sess, args.restore)
            start_epoch = int(basename(args.restore.split('-')[-1])) + 1
            global train_iter
            train_iter = start_epoch * len(train_set) // args.batch_size
            log("Start Epoch: ", start_epoch)
        else:
            start_epoch = 1
        train_writer = tf.summary.FileWriter(args.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(args.log_dir + '/test')

        print("Training")

        best_loss = None
        best_epoch = start_epoch - 1
        for epoch in range(start_epoch, args.epochs+1):
            start_time = time.time()
            log('Log: {} Discount Factor: {} Epochs w/o dec: {}'.format(
                folder_ind, get_discount_factor(), epoch-best_epoch-1))
            if args.decay_cutoff != -1 and epoch-best_epoch-1 >= args.decay_cutoff:
                lr_params['decay'] *= args.decay_factor
                log('Decaying learning rates to: {}'.format(get_learning_rates()))
                best_epoch = epoch
            if args.long:
                run_long_rollouts(sess)
            loss_, pos_loss_, aux_loss_ = run_epoch(sess, train_writer, train=True)
            log('TRAIN: Epoch: {} Total Loss: {:.6E} Pos Loss: {:.6E} Aux Loss: {:.6E} Time: {:.2f}s'.format(
                epoch, loss_, pos_loss_, aux_loss_, time.time() - start_time))

            loss_, pos_loss_, aux_loss_ = run_epoch(sess, test_writer, train=False)
            log('TEST: Total Loss: {:.6E} Pos Loss: {:.6E} Aux Loss: {:.6E} \n'.format(loss_, pos_loss_, aux_loss_))

            if args.baseline:
                break

            if best_loss is None or pos_loss_ < best_loss or args.save_all:
                best_loss = pos_loss_
                best_epoch = epoch
                saver.save(sess, join(args.ckpt_dir, 'model'), epoch)

if __name__ == '__main__':
    log("Arguments:", args)
    run()


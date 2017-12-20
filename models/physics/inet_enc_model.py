from __future__ import print_function, division

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, LSTMStateTuple

from shared import mlp

def relation_net(x, code_size):
    """RelationNet computes the net of pairwise interactions for each object. 

    @param x - input tensor [batch_size, n_objects, x_size]
    """

    n_objects = int(x.get_shape()[1])
    x_size = int(x.get_shape()[2])

    scramble_inds = [] 
    object_inds = list(range(n_objects))
    for _ in range(n_objects-1):
        object_inds = object_inds[-1:] + object_inds[:-1]
        scramble_inds.extend(object_inds)

    x_left = tf.tile(x, [1, n_objects - 1, 1])
    x_right = tf.stack([x[:,ind] for ind in scramble_inds], axis=1)

    x_pair = tf.concat([x_left, x_right], axis=2)

    x_pair_size = 2*x_size
    assert int(x_pair.get_shape()[2]) == x_pair_size

    h = tf.reshape(x_pair, [-1, x_pair_size])

    with tf.variable_scope("mlp"):
        h = mlp(h, [code_size, code_size, code_size])

    h = tf.reshape(h, [-1, n_objects-1, n_objects, code_size])
    h = tf.reduce_max(h, axis=1)
    h = tf.reshape(h, [-1, code_size])

    return h

def interaction_net(x, n_objects, code_size):
    """InteractionNet computes enc from.

    @param x - input tensor [batch_size * n_objects, x_size]
    @param n_objects - num objects
    @param code_size - code size
    """

    x_size = int(x.get_shape()[1])

    with tf.variable_scope("re_net"): # Relation Network
        pair_dynamics = relation_net(tf.reshape(x, [-1, n_objects, x_size]), code_size)

    with tf.variable_scope("sd_net"): # Self-Dynamics Network
        self_dynamics = mlp(x, [code_size, code_size, code_size])

    with tf.variable_scope("agg"): # Agg Network
        inp = tf.concat([x, pair_dynamics, self_dynamics], axis=1)
        enc = mlp(inp, [code_size, code_size, code_size])

    return enc

def inet_enc_net(enc_x, lstm_widths, dense_widths):
    """Computes an encoding vector using stacked LSTMs.
    @param enc_x: the input tensor of shape [batch_size, n_obs_frames, n_objects, state_size]
    @param lstm_widths: list of hidden layer widths in LSTM.
    @param dense_widths: list of hidden layer widths in dense layer.
    
    Final enc_size per object will be dense_widths[-1] // n_objects
    
    """

    n_obs_frames = int(enc_x.get_shape()[1])
    n_objects = int(enc_x.get_shape()[2])
    state_size = int(enc_x.get_shape()[3])

    states = tf.unstack(enc_x, axis=1)
    states = [tf.reshape(samp, [-1, state_size]) for samp in states]
    n_enc_window = 4
    code_size = 40

    enc = tf.get_variable("initial_enc", shape=(1, code_size), trainable=True)
    enc_shape = [tf.shape(states[0])[0], 1]
    enc = tf.tile(enc, enc_shape)

    for start_ind in range(n_obs_frames - n_enc_window + 1):
        cur_inp = tf.concat(states[start_ind:start_ind+n_enc_window] + [enc], axis=1)

        with tf.variable_scope("enc_inet", reuse=(start_ind != 0)):
            enc = interaction_net(cur_inp, n_objects, code_size)

    with tf.variable_scope("mlp"):
        enc = mlp(enc, [x / 3 for x in dense_widths])

    enc = tf.reshape(enc, [-1, n_objects, 1])

    return enc


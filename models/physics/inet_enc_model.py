from __future__ import print_function, division

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, LSTMStateTuple, RNNCell

from shared import mlp, dense

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

class SymGRUCell(RNNCell):
    def __init__(self, n_units_per_obj, n_objects):
        self.n_units_per_obj = n_units_per_obj
        self.n_objects = n_objects

    @property
    def state_size(self):
        return self.n_units_per_obj

    @property
    def output_size(self):
        return self.n_units_per_obj

    def __call__(self, inputs, state):
        """Takes in input tensor [batch_size * n_objects, input_size]
        state tensor [batch_size * n_objects, n_units_per_obj]
        """

        input_size = int(inputs.get_shape()[1])
        inputs_li = tf.unstack(tf.reshape(inputs, [-1, self.n_objects, input_size]), axis=1)
        states_li = tf.unstack(tf.reshape(state, [-1, self.n_objects, self.n_units_per_obj]), axis=1)
        inputs_te = inputs
        states_te = state

        sum_inputs_li = []
        sum_states_li = []
        for i in range(self.n_objects):
            sum_inputs_li.append(tf.add_n(inputs_li[:i] + inputs_li[i+1:]))
            sum_states_li.append(tf.add_n(states_li[:i] + states_li[i+1:]))

        sum_inputs_te = tf.reshape(tf.stack(sum_inputs_li, axis=1), [-1, input_size])
        sum_states_te = tf.reshape(tf.stack(sum_states_li, axis=1), [-1, self.n_units_per_obj])

        inp = tf.concat([inputs_te, states_te, sum_inputs_te, sum_states_te], axis=1)
        with tf.variable_scope("z"):
            z = dense(inp, self.n_units_per_obj, activation=tf.nn.sigmoid)

        with tf.variable_scope("r"):
            r = dense(inp, self.n_units_per_obj, activation=tf.nn.sigmoid)

        states_r_te = r * states_te
        states_r_li = tf.unstack(tf.reshape(states_r_te, [-1, self.n_objects, self.n_units_per_obj]), axis=1)
        sum_states_r_li = []
        for i in range(self.n_objects):
            sum_states_r_li.append(tf.add_n(states_r_li[:i] + states_r_li[i+1:]))
        sum_states_r_te = tf.reshape(tf.stack(sum_states_r_li, axis=1), [-1, self.n_units_per_obj])

        new_inp = tf.concat([inputs_te, sum_inputs_te, states_r_te, sum_states_r_te], axis=1)
        with tf.variable_scope("new"):
            new = dense(new_inp, self.n_units_per_obj, activation=tf.nn.tanh)

        h = z * states_te + (1 - z) * new

        return h, h

class GRUCell(RNNCell):
    def __init__(self, n_units):
        self.n_units = n_units

    @property
    def state_size(self):
        return self.n_units

    @property
    def output_size(self):
        return self.n_units

    def __call__(self, inputs, state):
        """Takes in input tensor [batch_size * n_objects, input_size]
        state tensor [batch_size * n_objects, n_units_per_obj]
        """

        input_size = int(inputs.get_shape()[1])
        inputs_te = inputs
        states_te = state

        inp = tf.concat([inputs_te, states_te], axis=1)
        with tf.variable_scope("z"):
            z = dense(inp, self.n_units, activation=tf.nn.sigmoid)

        with tf.variable_scope("r"):
            r = dense(inp, self.n_units, activation=tf.nn.sigmoid)

        states_r_te = r * states_te

        new_inp = tf.concat([inputs_te, states_r_te], axis=1)
        with tf.variable_scope("new"):
            new = dense(new_inp, self.n_units, activation=tf.nn.tanh)

        h = z * states_te + (1 - z) * new

        return h, h

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

    h = tf.transpose(enc_x, [1, 0, 2, 3])
    h = tf.reshape(h, [n_obs_frames, -1, n_objects * state_size])

    multi_rnn_cell = MultiRNNCell([SymGRUCell(width) for width in lstm_widths])

    h, state = tf.nn.dynamic_rnn(
            cell=multi_rnn_cell,
            time_major=True,
            inputs=h,
            scope="lstms",
            dtype=tf.float32)

    enc = h[-1] # Shape of [batch_size, lstm_widths[-1]]

    with tf.variable_scope("mlp"):
        enc = mlp(enc, [x for x in dense_widths // n_objects])

    enc = tf.reshape(enc, [-1, n_objects, dense_widths[-1] // n_objects])

    return enc


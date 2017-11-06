from __future__ import print_function, division

import tensorflow as tf
from tensorflow.contrib.rnn import static_rnn
from tensorflow.contrib.rnn import BasicLSTMCell

from shared import mlp

def lstm_enc_net(enc_x, lstm_widths, dense_widths):
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
    h = tf.unstack(h, axis=0)

    with tf.variable_scope("lstms"):
        for ind, width in enumerate(lstm_widths):
            lstm_cell = BasicLSTMCell(width)
            h, state = static_rnn(lstm_cell, h,
                    scope="lstm_"+str(ind),
                    dtype=tf.float32)

    enc = h[-1] # Shape of [batch_size, enc_widths[-1]]

    with tf.variable_scope("mlp"):
        enc = mlp(enc, dense_widths)

    enc = tf.reshape(enc, [-1, n_objects, dense_widths[-1] // n_objects])

    return enc


from __future__ import print_function, division

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, LSTMStateTuple

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

    n_enc_window = 4
    hs = []
    for i in range(n_enc_window-1):
        hs.append(h[i:-(n_enc_window-i-1)])
    hs.append(h[n_enc_window-1:])
    h = tf.concat(hs, axis=2)

    multi_rnn_cell = MultiRNNCell([LSTMCell(width) for width in lstm_widths])

    # trainable initial state
    tile_size = tf.concat([tf.shape(h)[1:2], tf.constant([1], dtype=tf.int32)], axis=0)
    initial_state = tuple([LSTMStateTuple(*[tf.tile(tf.get_variable(
                "initial_state_"+str(i)+'_'+str(j),
                shape=(1,width),
                initializer=tf.zeros_initializer()
                ), tile_size)
            for j, width in enumerate(state_size)])
        for i, state_size in enumerate(multi_rnn_cell.state_size)])

    h, state = tf.nn.dynamic_rnn(
            cell=multi_rnn_cell,
            time_major=True,
            initial_state=initial_state,
            inputs=h,
            scope="lstms",
            dtype=tf.float32)

    enc = h[-1] # Shape of [batch_size, lstm_widths[-1]]

    with tf.variable_scope("mlp"):
        enc = mlp(enc, dense_widths)

    enc = tf.reshape(enc, [-1, n_objects, dense_widths[-1] // n_objects])

    return enc


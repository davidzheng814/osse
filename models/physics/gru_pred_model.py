import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, LSTMStateTuple, RNNCell

from shared import mlp, dense

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


def gru_pred_net(x0, enc, frames_per_samp, code_size, n_ro_frames, offsets,
        noise_ratio=0.0):
    """Returns list of n_ro_frames future object states from an initial object state.
   
    @param x0 - initial state tensor of shape [batch_size, n_prep_frames, n_objects, state_size]
    @param enc - encoding tensor of shape [batch_size, n_objects, enc_size]
    @param noise_ratio - the amount of white noise to add to states during rollout.
        The noise is distributed in a gaussian with standard deviation equal to
        noise_ratio * batch_std(states)
    
    """

    n_objects = int(x0.get_shape()[2])
    state_size = int(x0.get_shape()[3])
    enc_size = int(enc.get_shape()[2])

    enc = tf.reshape(enc, [-1, enc_size])

    orig_x0 = x0
    x0 = tf.unstack(x0, axis=1)
    x0 = [tf.reshape(state, [-1, state_size]) for state in x0]
    n_prep_frames = len(x0)

    lstm_widths = [64, 64, 64, 64]
    multi_rnn_cell = MultiRNNCell([SymGRUCell(width, n_objects) for width in lstm_widths])

    state_mean, state_var = tf.nn.moments(orig_x0, [0, 1, 2])
    state_std = tf.unstack(tf.sqrt(state_var))

    pred_states = x0[:1]
    cell_state = multi_rnn_cell.zero_state(tf.shape(x0[0])[0], tf.float32)
    for frame_ind in range(1, n_ro_frames):
        inp = pred_states[frame_ind-1] if frame_ind > n_prep_frames else x0[frame_ind-1]
        inp = tf.concat([inp, enc], axis=1)
        pred_code, cell_state  = multi_rnn_cell(inp, cell_state)

        with tf.variable_scope("code_to_state", reuse=tf.AUTO_REUSE):
            pred_state = mlp(pred_code, [64, state_size])

            noise = [tf.random_normal(shape=tf.shape(pred_state)[:-1], mean=0.0,
                stddev=noise_ratio * element_std) for element_std in state_std]
            noise = tf.stack(noise, axis=1)
            pred_state += noise
            pred_states.append(pred_state)

    pred_states = tf.reshape(tf.stack(pred_states, axis=1), [-1, n_objects, n_ro_frames, state_size])
    pred_states = tf.transpose(pred_states, [0, 2, 1, 3])

    aux_loss = tf.zeros([])

    return pred_states, aux_loss


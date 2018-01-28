import tensorflow as tf
import json
import numpy as np
from os.path import join
from sklearn.linear_model import LinearRegression

def dense(x, out_width, activation=None):
    inp_width = int(x.get_shape()[1])
    weight = tf.get_variable("weight",
            shape=(inp_width, out_width))
    bias = tf.get_variable("bias",
            shape=(out_width,))
    h = tf.matmul(x, weight) + bias
    if activation is not None:
        h = activation(h)

    return h

def mlp(x, widths, relu_final=False):
    """Multi-layer perceptron.

    @param x - input tensor [batch_size, inp_size]
    @param widths - hidden layer widths
    """

    h = x
    for ind, out_width in enumerate(widths):
        with tf.variable_scope("dense_"+str(ind)):
            if relu_final or ind != len(widths) - 1:
                h = dense(h, out_width, activation=tf.nn.relu)
            else:
                h = dense(h, out_width)

    return h

def log(log_dir, *text):
    print(*text)
    with open(join(log_dir, 'log.txt'), 'a') as f:
        f.write(' '.join([str(x) for x in text]) + '\n')

def get_enc_analysis(enc_pred, y_true):
    """
        @param enc_pred: [batch_size, n_objects, enc_size]
        @param y_true: [batch_size, n_objects * y_size]
    """
    r2s = []
    for obj_ind in range(enc_pred.shape[1]):
        X = enc_pred[:,obj_ind]
        y = y_true[:,obj_ind]

        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        r2s.append(r2)

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

import tensorflow as tf

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


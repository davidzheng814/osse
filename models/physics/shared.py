import tensorflow as tf

def mlp(x, widths, relu_final=False):
    """Multi-layer perceptron.

    @param x - input tensor [batch_size, inp_size]
    @param widths - hidden layer widths
    """

    inp_widths = [int(x.get_shape()[1])] + widths[:-1]

    h = x
    for ind, (inp_width, out_width) in enumerate(zip(inp_widths, widths)):
        with tf.variable_scope("dense_"+str(ind)):
            weight = tf.get_variable("weight",
                    shape=(inp_width, out_width))
            bias = tf.get_variable("bias",
                    shape=(out_width,))

            h = tf.matmul(h, weight) + bias
            if relu_final or ind != len(widths) - 1:
                h = tf.nn.relu(h)

    return h

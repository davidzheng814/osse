import tensorflow as tf

from shared import mlp

def relation_net(x, re_widths, effect_width):
    """RelationNet computes the net of pairwise interactions for each object. 

    @param x - input tensor [batch_size, n_objects, full_state_size]
    """

    n_objects = int(x.get_shape()[1])
    full_state_size = int(x.get_shape()[2])

    scramble_inds = [] 
    object_inds = list(range(n_objects))
    for _ in range(n_objects-1):
        object_inds = object_inds[-1:] + object_inds[:-1]
        scramble_inds.extend(object_inds)

    x_left = tf.tile(x, [1, n_objects - 1, 1])
    x_right = tf.stack([x[:,ind] for ind in scramble_inds], axis=1)
    x_pair = tf.concat([x_left, x_right], axis=2)

    h = tf.reshape(x_pair, [-1, 2*full_state_size])

    with tf.variable_scope("mlp"):
        h = mlp(h, re_widths+[effect_width])

    h = tf.reshape(h, [-1, n_objects-1, n_objects, effect_width])
    h = tf.reduce_sum(h, axis=1)
    h = tf.reshape(h, [-1, effect_width])

    return h

def interaction_net(x, n_objects, re_widths, sd_widths, agg_widths, effect_width, out_width):
    """InteractionNet computes new object codes from previous object codes.

    @param x - input tensor [-1, full_state_size]
    @param n_objects - num objects
    """

    full_state_size = int(x.get_shape()[1])

    with tf.variable_scope("re_net"): # Relation Network
        pair_dynamics = relation_net(tf.reshape(x, [-1, n_objects, full_state_size]), re_widths, effect_width)

    with tf.variable_scope("sd_net"): # Self-Dynamics Network
        self_dynamics = mlp(x, sd_widths + [effect_width])

    with tf.variable_scope("effects"):
        effects = pair_dynamics + self_dynamics

    with tf.variable_scope("agg"): # Aggregation Network
        inp = tf.concat([x, effects], axis=1)
        pred = mlp(inp, agg_widths + [out_width]) # Hardcoded desired output

    return pred

def predict_net(ro_x_inp, enc_pred, n_ro_frames, re_widths, sd_widths, agg_widths, effect_width,
                out_width, noise_ratio=0.0):
    """Returns list of n_ro_frames future object states from an initial object state.
   
    @param ro_x_inp - initial state tensor of shape [batch_size, n_objects, state_size]
    @param enc_pred - encoding tensor of shape [batch_size, n_objects, enc_size]
    @param noise_ratio - the amount of white noise to add to states during rollout.
        The noise is distributed in a gaussian with standard deviation equal to
        noise_ratio * batch_std(states)
    
    """

    n_objects = int(ro_x_inp.get_shape()[1])
    state_size = int(ro_x_inp.get_shape()[2])
    enc_size = int(enc_pred.get_shape()[2])

    enc_pred = tf.reshape(enc_pred, [-1, enc_size]) # [batch_size*n_objects, enc_size]
    ro_x_inp = tf.reshape(ro_x_inp, [-1, state_size]) # [batch_size*n_objects, state_size]

    ro_x_preds = [] # all pred frames
    for i in range(1, n_ro_frames):
        if noise_ratio > 0.:
            state_mean, state_var = tf.nn.moments(ro_x_inp, [0])
            state_std = tf.unstack(tf.sqrt(state_var))
            noise = [tf.random_normal(shape=tf.shape(ro_x_inp)[:-1], mean=0.0,
                stddev=noise_ratio * element_std) for element_std in state_std]
            noise = tf.stack(noise, axis=1)
            ro_x_inp += noise

        full_state = tf.concat([ro_x_inp, enc_pred], axis=1)

        ro_x_pred = interaction_net(full_state, n_objects, re_widths, sd_widths, agg_widths,
                                    effect_width, out_width)
        ro_x_preds.append(ro_x_pred)
        ro_x_inp = ro_x_pred

    ro_x_pred = tf.stack([tf.reshape(x, [-1, n_objects, out_width]) for x in ro_x_preds], axis=1)

    return ro_x_pred


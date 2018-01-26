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
    reg_loss = tf.nn.l2_loss(h)
    h = tf.reduce_sum(h, axis=1)
    h = tf.reshape(h, [-1, effect_width])

    return h, reg_loss

def interaction_net(x, n_objects, re_widths, sd_widths, agg_widths, effect_width, out_width):
    """InteractionNet computes new object codes from previous object codes.

    @param x - input tensor [-1, full_state_size]
    @param n_objects - num objects
    """

    full_state_size = int(x.get_shape()[1])

    with tf.variable_scope("re_net"): # Relation Network
        pair_dynamics, reg_loss = relation_net(tf.reshape(x, [-1, n_objects, full_state_size]), re_widths, effect_width)

    with tf.variable_scope("sd_net"): # Self-Dynamics Network
        self_dynamics = mlp(x, sd_widths + [effect_width])

    with tf.variable_scope("effects"):
        tot_dynamics = pair_dynamics + self_dynamics

    with tf.variable_scope("agg"): # Aggregation Network
        inp = tf.concat([x, tot_dynamics], axis=1)
        pred = mlp(inp, agg_widths + [out_width]) # Hardcoded desired output

    return pred, reg_loss

def inet_enc_net(enc_x, re_widths, sd_widths, agg_widths, effect_width,
                enc_size, num_pred_frames):
    """Returns list of n_ro_frames future object states from an initial object state.
   
    @param enc_x - the input tensor of shape [batch_size, n_obs_frames, n_objects, state_size]
    @param enc_pred - encoding tensor of shape [batch_size, n_objects, enc_size]
    
    """

    n_obs_frames = int(enc_x.get_shape()[1])
    n_objects = int(enc_x.get_shape()[2])
    state_size = int(enc_x.get_shape()[3])

    enc = tf.zeros([tf.shape(enc_x)[0] * n_objects, enc_size])

    reg_loss = tf.zeros([])
    enc_x_li = tf.unstack(enc_x, axis=1)
    for i in range(n_obs_frames - num_pred_frames):
        ro_x_inp = tf.concat(enc_x_li[i:i+num_pred_frames], axis=2)
        ro_x_inp = tf.reshape(ro_x_inp, [-1, state_size*num_pred_frames]) # [batch_size*n_objects, state_size]
        full_state = tf.concat([ro_x_inp, enc], axis=1)

        enc, reg_loss_ = interaction_net(full_state, n_objects, re_widths, sd_widths, agg_widths,
                                    effect_width, enc_size)
        reg_loss += reg_loss_

    reg_loss /= n_obs_frames # normalize after summing
    enc = tf.reshape(enc, [-1, n_objects, enc_size])

    return enc, reg_loss

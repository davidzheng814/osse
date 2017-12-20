import tensorflow as tf

from shared import mlp

def relation_net(x, code_size):
    """RelationNet computes the net of pairwise interactions for each object. 

    @param x - input tensor [batch_size, n_objects, code_size]
    """

    n_objects = int(x.get_shape()[1])
    code_size = int(x.get_shape()[2])

    scramble_inds = [] 
    object_inds = list(range(n_objects))
    for _ in range(n_objects-1):
        object_inds = object_inds[-1:] + object_inds[:-1]
        scramble_inds.extend(object_inds)

    x_left = tf.tile(x, [1, n_objects - 1, 1])
    x_right = tf.stack([x[:,ind] for ind in scramble_inds], axis=1)

    x_pair = tf.concat([x_left, x_right], axis=2)

    assert int(x_pair.get_shape()[2]) == 2*code_size

    h = tf.reshape(x_pair, [-1, 2*code_size])

    with tf.variable_scope("mlp"):
        h = mlp(h, [code_size, code_size, code_size])

    h = tf.reshape(h, [-1, n_objects-1, n_objects, code_size])
    h = tf.reduce_sum(h, axis=1)
    h = tf.reshape(h, [-1, code_size])

    return h

def interaction_net(x, n_objects, code_size):
    """InteractionNet computes new object codes from previous object codes.

    @param x - input tensor [batch_size * n_objects, code_size]
    @param n_objects - num objects
    """

    code_size = int(x.get_shape()[1])

    with tf.variable_scope("re_net"): # Relation Network
        pair_dynamics = relation_net(tf.reshape(x, [-1, n_objects, code_size]), code_size)

    with tf.variable_scope("sd_net"): # Self-Dynamics Network
        self_dynamics = mlp(x, [code_size, code_size])

    with tf.variable_scope("aff"): # Aff Network
        tot_dynamics = pair_dynamics + self_dynamics
        tot_dynamics = mlp(tot_dynamics, [code_size, code_size, code_size])

    with tf.variable_scope("agg"): # Aggregation Network
        inp = tf.concat([x, tot_dynamics], axis=1)
        pred = mlp(inp, [code_size, code_size, code_size])

    return pred

def predict_net_cell(x, n_offsets, n_objects, code_size, state_size):
    """Combines object codes at multiple time steps to compute new object codes. 

    @param x - list of n_offsets input tensors, with shapes:
               [batch_size * n_objects, code_size]
    @param n_offsets - number of timestep offsets to use. 
    """

    code_size = int(x[0].get_shape()[1])

    preds = [] # list of preds, with shapes [batch_size * n_objects, code_size]
    for ind in range(n_offsets):
        with tf.variable_scope("inet_"+str(ind)):
            pred = interaction_net(x[ind], n_objects, code_size)
            preds.append(pred)

    h = tf.concat(preds, axis=1)
    with tf.variable_scope("inets_agg"):
        h = mlp(h, [code_size, code_size, code_size])

    return h

def predict_net(x0, enc, frames_per_samp, code_size, n_ro_frames, offsets,
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

    codes = [None for _ in range(frames_per_samp-1)]
    aux_losses = []
    for frame_ind in range(frames_per_samp-1, n_prep_frames):
        full_state = tf.concat(x0[frame_ind-frames_per_samp+1:frame_ind+1]+[enc], axis=1,
                name="full_state_"+str(frame_ind))

        with tf.variable_scope("state_to_code", reuse=(frame_ind != frames_per_samp-1)):
            code = mlp(full_state, [code_size])

        codes.append(code)

        with tf.variable_scope("code_to_state", reuse=(frame_ind != frames_per_samp-1)):
            aux_state = mlp(code, [state_size])

        true_state = x0[frame_ind]
        aux_losses.append(tf.reduce_mean(tf.squared_difference(true_state, aux_state)))

    aux_loss = tf.reduce_mean(tf.stack(aux_losses))

    states = [state for state in x0]
    state_mean, state_var = tf.nn.moments(orig_x0, [0, 1, 2])
    state_std = tf.unstack(tf.sqrt(state_var))
    for frame_ind in range(n_prep_frames, n_ro_frames):
        cur_inp = [codes[frame_ind-offset] for offset in offsets]

        with tf.variable_scope("pred_net_cell", reuse=(frame_ind != n_prep_frames)):
            pred_code = predict_net_cell(cur_inp, len(offsets), n_objects, code_size, state_size)

        with tf.variable_scope("code_to_state", reuse=True):
            pred_state = mlp(pred_code, [state_size])
            noise = [tf.random_normal(shape=tf.shape(pred_state)[:-1], mean=0.0,
                stddev=element_std) for element_std in state_std]
            noise = tf.stack(noise, axis=1)
            pred_state += noise
            states.append(pred_state)
            full_state = tf.concat(states[-frames_per_samp:]+[enc], axis=1)

        with tf.variable_scope("state_to_code", reuse=True):
            codes.append(mlp(full_state, [code_size]))

    states = tf.reshape(tf.stack(states, axis=1), [-1, n_objects, n_ro_frames, state_size])
    states = tf.transpose(states, [0, 2, 1, 3])

    return states, aux_loss


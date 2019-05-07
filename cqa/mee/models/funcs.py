import tensorflow as tf


def instant_denses(t, uas, name, w_init=None, w_reg=None):
    for i, (u, a) in enumerate(uas):
        t = tf.layers.dense(
            t, u, activation=a, name='{}_{}'.format(name, i),
            kernel_initializer=w_init, kernel_regularizer=w_reg,
        )
    return t


def postpone_denses(denses, t, name):
    for i, d in enumerate(denses):
        with tf.name_scope(name + '_{}'.format(i)):
            t = d.apply(t)
    return t


def get_kl_loss(true_probs, pred_probs):
    assert len(true_probs.shape) == len(pred_probs.shape)
    import tensorflow.distributions as tfd
    true = tfd.Categorical(probs=true_probs + 1e-9)
    pred = tfd.Categorical(probs=pred_probs + 1e-9)
    pointwise_diverge = tfd.kl_divergence(true, pred, allow_nan_stats=False)
    kl_loss = tf.reduce_sum(pointwise_diverge)
    return kl_loss


def mask_diagly(t, name, value=-1e9):
    with tf.name_scope(name):
        return t + value * tf.eye(num_rows=tf.shape(t)[0])


def pairwise_sub(t, name):
    assert len(t.shape) == 1
    with tf.name_scope(name):
        t_0 = tf.expand_dims(t, axis=0)
        t_1 = tf.expand_dims(t, axis=1)
        sub = tf.subtract(t_1, t_0)
    return sub

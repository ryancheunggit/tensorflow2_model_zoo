import tensorflow as tf


def smooth_l1_loss(labels, predictions):
    """Smooth L1 Loss

    Use squared loss when difference is wihtin 1, otherwise use an L1 loss

    reference: http://arxiv.org/abs/1504.08083
    """
    n = tf.cast(tf.reduce_prod(tf.shape(labels)), dtype=tf.float32)
    l1_mask = tf.abs(labels - predictions) >= 1
    l2_mask = tf.abs(labels - predictions) < 1
    z1 = tf.abs(tf.boolean_mask(labels, l1_mask) - tf.boolean_mask(predictions, l1_mask)) - .5
    z2 = .5 * tf.square(tf.boolean_mask(labels, l2_mask) - tf.boolean_mask(predictions, l2_mask))
    loss = (tf.reduce_sum(z1) + tf.reduce_sum(z2)) / n
    return loss


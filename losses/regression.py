import tensorflow as tf


def smooth_l1_loss(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    error = y_true - y_pred
    abs_error = tf.abs(error)
    mask = abs_error < 1
    squared_loss = tf.square(error)
    linear_loss = abs_error - .5
    loss = tf.where(mask, squared_loss, linear_loss)
    return tf.reduce_mean(loss, 1)


class SmoothL1Loss(tf.keras.losses.Loss):
    """Smooth L1 Loss

    Use squared loss when difference is wihtin 1, otherwise use an L1 loss.
    It turned out is is just [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) with delta = 1.
    Anyway, implmented as an exercise.

    reference: http://arxiv.org/abs/1504.08083
    """
    def __init__(self,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name='smooth_l1_loss'):
        super(SmoothL1Loss, self).__init__(name=name, reduction=reduction)

    def call(self, y_true, y_pred):
       return smooth_l1_loss(y_true, y_pred)

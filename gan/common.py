import tensorflow as tf


class PixelwiseNormalization(tf.keras.Model):
    """Pixelwise feature normalization.

    reference: https://arxiv.org/abs/1710.10196
    > we normalize the feature vector in each pixel to unit length in the generator after each convolutional layer.
    """
    def __init__(self, data_format='channels_first', epsilon=1e-8, name='pixelwise_norm'):
        super(PixelwiseNormalization, self).__init__(name=name)
        self.data_format = data_format
        self.channel_axis = 1 if data_format == 'channels_first' else -1
        self.epsilon = epsilon

    def call(self, x):
        y = tf.sqrt(tf.reduce_mean(tf.square(x), axis=self.channel_axis, keepdims=True) + self.epsilon)
        return x / y


class MinibatchStdDev(tf.keras.Model):
    """Minibatch standard deviation.

    reference: https://arxiv.org/abs/1710.10196
    > ... by adding a minibatch layer towards the end of the discriminator, where the layer learns a large tensor
        that projects the input activation to an array of statistics. A separate set of statistics is produced
        for each example in a minibatch and it is concatenated to the layer's output.
    """
    def __init__(self, data_format='channels_first', group_size=4, epsilon=1e-8, name='minibatch_std_dev'):
        super(MinibatchStdDev, self).__init__(name=name)
        self.data_format = data_format
        self.group_size = group_size
        self.epsilon = epsilon

    def call(self, x, epsilon=1e-8):
        if self.data_format == 'channels_first':
            batch_size, n_channels, height, width = tf.shape(x)
        else:
            batch_size, height, width, n_channels = tf.shape(x)
            x = tf.transpose(x, perm=[0, 3, 1, 2])
        group_size = tf.minimum(self.group_size, batch_size)
        assert int(batch_size) % int(group_size) == 0, 'batch_size {} must be divisible by group size {}'.format(
                batch_size, group_size)

        y = tf.reshape(x, [group_size, -1, n_channels, height, width])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + self.epsilon)
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)
        y = tf.cast(y, x.dtype)
        y = tf.tile(y, [group_size, 1, height, width])
        xx = tf.concat([x, y], axis=1)
        if self.data_format == 'channels_last':
            xx = tf.transpose(xx, perm=[0, 2, 3, 1])
        return xx


def _test_pixelwise_norm():
    x = tf.random.uniform((32, 3, 64, 64))
    m1 = PixelwiseNormalization(data_format='channels_first', epsilon=1e-8)
    o1 = m1(x)
    m2 = PixelwiseNormalization(data_format='channels_last', epsilon=1e-8)
    o2 = tf.transpose(m2(tf.transpose(x, perm=[0, 2, 3, 1])), perm=[0, 3, 1, 2])
    assert tf.reduce_all(tf.abs(o1 - o2) <= 1e-7)


def _test_minibatch_stddev():
    x = tf.random.uniform((32, 3, 64, 64))
    m1 = MinibatchStdDev(data_format='channels_first', epsilon=1e-8)
    o1 = m1(x)
    m2 = MinibatchStdDev(data_format='channels_last', epsilon=1e-8)
    o2 = tf.transpose(m2(tf.transpose(x, perm=[0, 2, 3, 1])), perm=[0, 3, 1, 2])
    assert tf.reduce_all(tf.abs(o1 - o2) <= 1e-7)



if __name__ == '__main__':
    _test_pixelwise_norm()
    _test_minibatch_stddev()

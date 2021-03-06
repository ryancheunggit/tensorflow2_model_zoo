__all__ = ['Conv2d', 'MaxPool2d', 'SEBlock', 'Flatten', 'GeMPooling', 'get_num_params']

import math
import numpy as np
import tensorflow as tf


class Conv2d(tf.keras.Model):
    """Wrapper class for 2D convolution layers.

    Encapsulate three types of convolution layers, and handle padding explicitly.

    Arguments:
        in_channels: int, number of input channels.
        out_channels: int, number of output channels.
        kernel_size: int or tuple of 2 ints, size of the convolution filters.
        strides: int or tuple of 2 ints, strides values.
        padding: int or tuple of 2 ints, number of 0s padding to each side of height and width,
            eg: padding = (2, 3):
            this will pad 2 rows of zeros to top, 2 rows of zeros to bottom,
            and 3 columns of zeros to left, 3 columns of zeros to right to each input.
        dilation: int or tuple of 2 ints, dilation rate for dilated convolution.
        groups: int, controls the connection between inputs and outputs.
            currently supports three types, see source for detail.
        use_bias: Boolean, whether the layer uses a bias vector.
        data_format: str, either 'channels_first' or 'channels_last'.
        kernel_initializer: Kernel initializer object or string name for initializer,
            tf.keras.layers.Conv2D defaults to 'glorot_uniform', which I found not working well
            from time to time.
        name: str, name of this layer.
    """
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            strides=1,
            padding=0,
            dilation=1,
            groups=1,
            use_bias=True,
            data_format='channels_last',
            kernel_initializer=None,
            name='conv2d'):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.kernel_size = kernel_size
        strides = (strides, strides) if isinstance(strides, int) else strides
        self.strides = strides
        padding = (padding, padding) if isinstance(padding, int) else padding
        self.padding = padding
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.data_format = data_format
        if not kernel_initializer:
            self.kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
        else:
            self.kernel_initializer = kernel_initializer
        self._name = name

        if groups == 1:
            self.mode = 'fulldepth'
            self.conv = tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding='valid',
                data_format=data_format,
                dilation_rate=dilation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                name=name
            )
        elif (groups == in_channels == out_channels):
            self.mode = 'depthwise'
            assert all(hop_length == 1 for hop_length in dilation)
            self.conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=strides,
                padding='valid',
                depth_multiplier=1,
                data_format=data_format,
                use_bias=use_bias,
                depthwise_initializer=kernel_initializer,
                name=name
            )
        else:
            self.mode = 'grouped'
            assert (in_channels % groups == 0)
            assert (out_channels % groups == 0)
            self.in_group_channels = in_channels // groups
            self.out_group_channels = out_channels // groups
            self.convs = [
                tf.keras.layers.Conv2D(
                    filters=self.out_group_channels,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='valid',
                    data_format=data_format,
                    dilation_rate=dilation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    name=name + '/conv_group_{}'.format(i)
                )
                for i in range(groups)
            ]
            concat_axis = 1 if self.data_format == 'channels_first' else -1
            self.concat = tf.keras.layers.Concatenate(axis=concat_axis, name=name + '/concat')

    def call(self, x):
        if any(margin > 0 for margin in self.padding):
            if self.data_format == 'channels_first':
                paddings = [[0, 0], [0, 0], list(self.padding), list(self.padding)]
            else:
                paddings = [[0, 0], list(self.padding), list(self.padding), [0, 0]]
            x = tf.pad(x, paddings=paddings)

        if self.mode in ['fulldepth', 'depthwise']:
            x = self.conv(x)
        else:
            convs_out = []
            for i in range(self.groups):
                if self.data_format == 'channels_first':
                    xi = x[:, i * self.in_group_channels: (i + 1) * self.in_group_channels, :, :]
                else:
                    xi = x[:, :, :, i * self.in_group_channels: (i + 1) * self.in_group_channels]
                convs_out.append(self.convs[i](xi))
            x = self.concat(convs_out)
        return x


class MaxPool2d(tf.keras.Model):
    def __init__(self,
            pool_size,
            strides,
            padding=0,
            ceil_mode=False,
            data_format='channels_last',
            name='maxpool2d'):
        """Max Pooling 2D layer with explicit handling of paddings.

        Arguments:
            pool_size: int or tuple of 2 ints, downsample scale.
            strides: int or tuple of 2 ints, strides values.
            padding: int or tuple of 2 ints, reflective padding applied to each dimension.
                could be modified by +1 for certain cases in 'ceil_mode = True'.
            ceil_mode: Boolean, whether to use ceil rather than floor when calculating output shape.
            data_format: str, either 'channels_first' or 'channels_last'.
            name: str, name of this layer.
        """
        super(MaxPool2d, self).__init__()
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.strides = (strides, strides) if isinstance(strides, int) else strides
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.ceil_mode = ceil_mode
        self.data_format = data_format
        self._name = name
        self.pool = tf.keras.layers.MaxPool2D(
            pool_size=pool_size,
            strides=strides,
            padding='valid',
            data_format=data_format,
            name=name
        )

    def call(self, x, training=False):
        height, width = (x.shape[2], x.shape[3]) if self.data_format == 'channels_first' else (x.shape[1], x.shape[2])
        padding = self.padding
        if self.ceil_mode:
            out_height = float(height + 2 * self.padding[0] - self.pool_size[0]) / self.strides[0] + 1.0
            out_width = float(width + 2 * self.padding[1] - self.pool_size[1]) / self.strides[1] + 1.0
            if math.ceil(out_height) > math.floor(out_height):
                padding = (padding[0] + 1, padding[1])
            if math.ceil(out_width) > math.floor(out_width):
                padding = (padding[0], padding[1] + 1)

        if any(margin > 0 for margin in padding):
            if self.data_format == 'channels_first':
                x = tf.pad(x, [[0, 0], [0, 0], list(padding), list(padding)], mode='REFLECT')
            else:
                x = tf.pad(x, [[0, 0], list(padding), list(padding), [0, 0]], mode='REFLECT')
        x = self.pool(x)
        return x


class SEBlock(tf.keras.Model):
    """Squeeze-and-Excitation block.

       x_in -----------------
    (bsxhxwxc)              |
         |                  |
         |          global average pool     (squeeze global spatial information into channel descriptor)
         |             (bsx1x1xc)
         |                  |
         |         fc (conv1x1) + relu
         |             (bsx1x1xc/r)
         |                   |              (excitation)
         |         fc (conv1x1) + sigmoid
         |              (bsx1x1xc)
         |                   |
         *--------------------      (channel wise multiplication)
         |
       x_out       (self attention)
     (bsxhxwxc)

    reference: https://arxiv.org/abs/1709.01507.
    """
    def __init__(self, channels, reduction, data_format='channels_last', name='se_block'):
        super(SEBlock, self).__init__(name=name)
        self.data_format = 'NHWC' if data_format == 'channels_last' else 'NCHW'
        mid_channels = channels // reduction
        self.fc1 = Conv2d(
            in_channels=channels, out_channels=mid_channels, kernel_size=1, strides=1, use_bias=True,
            data_format=data_format, name=name + '/squeeze')
        self.fc2 = Conv2d(
            in_channels=mid_channels, out_channels=channels, kernel_size=1, strides=1, use_bias=True,
            data_format=data_format, name=name + '/excitation')

    def call(self, x, training=False):
        input_shape = tf.shape(x)
        h, w = input_shape[1:3] if self.data_format == 'channels_last' else input_shape[2:4]
        s = tf.nn.avg_pool2d(x, ksize=(h, w), strides=None, padding='SAME', data_format=self.data_format)
        s = self.fc1(s)
        s = tf.keras.activations.relu(s)
        s = self.fc2(s)
        s = tf.keras.activations.sigmoid(s)
        return s * x


class Flatten(tf.keras.Model):
    """Flatten layer.

    Argument:
        data_format: str, either 'channels_first' or 'channels_last'
    """
    def __init__(self, data_format='channels_last'):
        super(Flatten, self).__init__()
        self.data_format = data_format

    def call(self, x):
        if self.data_format == 'channels_last':
            x = tf.transpose(x, perm=(0, 3, 1, 2))
        x = tf.reshape(x, shape=(-1, np.prod(x.get_shape().as_list()[1:])))
        return x


class ChannelShuffle(tf.keras.layers.Layer):
    """Channel Shuffle operation from shufflenet.

    for example:
        input with 4 channels c1,c2,c3,c4, with n_groups == 2, shuffle result is: c1,c3,c2,c4
        input with 6 channels c1,c2,c3,c4,c5,c6 with n_groups == 2, shuffle result is: c1,c4,c2,c5,c3,c6
        input with 6 channels c1,c2,c3,c4,c5,c6 with n_groups == 3, shuffle result is: c1,c3,c5,c2,c4,c6
    """
    def __init__(self, data_format='channels_last', n_groups=2, name='channel_shuffle'):
        super(ChannelShuffle, self).__init__(name=name)
        self.data_format = data_format
        self.n_groups = n_groups

    def call(self, x):
        shape = x.get_shape().as_list()
        if self.data_format == 'channels_last':
            height, width, channels = shape[1:4]
        else:
            channels, height, width = shape[1:4]
        assert channels % self.n_groups == 0
        channels_per_group = channels // self.n_groups
        if self.data_format == 'channels_last':
            x = tf.reshape(x, shape=(-1, height, width, self.n_groups, channels_per_group))
            x = tf.transpose(x, perm=(0, 1, 2, 4, 3))
            x = tf.reshape(x, shape=(-1, height, width, channels))
        else:
            x = tf.reshape(x, shape=(-1, self.n_groups, channels_per_group, height, width))
            x = tf.transpose(x, perm=(0, 2, 1, 3, 4))
            x = tf.reshape(x, shape=(-1, channels, height, width))
        return x


class GeMPooling(tf.keras.layers.Layer):
    """Generalized mean pooling.

    reference: https://arxiv.org/abs/1711.02512
    """
    def __init__(self, p=3, epsilon=1e-6, data_format='channels_last', name='gempooling'):
        super(GeMPooling, self).__init__(name=name)
        self.p = p
        self.epsilon = epsilon
        self.data_format=data_format

    def call(self, x):
        if self.data_format == 'channels_first':
            x = tf.transpose(x, perm=(0, 2, 3, 1))

        N, H, W, C = tf.shape(x)
        data_format = 'NHWC'
        ksize = (1, H, W, 1)

        x = tf.where(x < self.epsilon, self.epsilon, x)
        x = tf.pow(x, self.p)
        x = tf.nn.avg_pool2d(x, ksize=ksize, data_format=data_format, strides=1, padding='VALID')
        x = tf.pow(x, 1 / self.p)

        if self.data_format == 'channels_first':
            x = tf.transpose(x, perm=(0, 3, 1, 2))
        return x



def get_num_params(module):
    """Calculate the number of parameters of a neural network module."""
    return np.sum([np.prod(v.get_shape().as_list()) for v in module.trainable_variables])


def _test_Conv2d():
    x = tf.random.uniform((32, 24, 24, 16))
    conv1 = Conv2d(in_channels=16, out_channels=40, kernel_size=3, strides=2, padding=0, dilation=1, groups=1)
    out1 = conv1(x)
    assert get_num_params(conv1) == (16 * 3 * 3 + 1) * 40
    assert out1.shape == (32, 11, 11, 40)
    conv2 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, strides=2, padding=0, dilation=1, groups=16)
    out2 = conv2(x)
    assert get_num_params(conv2) == (1 * 3 * 3 + 1) * 16
    assert out2.shape == (32, 11, 11, 16)
    conv3 = Conv2d(in_channels=16, out_channels=40, kernel_size=3, strides=2, padding=0, dilation=1, groups=4)
    out3 = conv3(x)
    assert get_num_params(conv3) == (16 / 4 * 3 * 3 + 1) * (40 / 4) * 4
    assert out3.shape == (32, 11, 11, 40)


def _test_MaxPool2d():
    x = np.expand_dims(np.expand_dims(np.array([[i] * 5 for i in range(5)]), 0), -1)
    x = tf.convert_to_tensor(x)
    pool1 = MaxPool2d(pool_size=2, strides=2, ceil_mode=False)
    out1 = pool1(x)
    assert out1.shape == (1, 2, 2, 1)
    pool2 = MaxPool2d(pool_size=2, strides=2, ceil_mode=True)
    out2 = pool2(x)
    assert out2.shape == (1, 3, 3, 1)


def _test_Flatten():
    x1 = tf.random.uniform((32, 3, 24, 24))
    flatten1 = Flatten(data_format='channels_first')
    out1 = flatten1(x1)
    assert out1.shape == (32, 1728)
    x2 = tf.random.uniform((32, 24, 24, 3))
    flatten2 = Flatten(data_format='channels_last')
    out2 = flatten2(x2)
    assert out2.shape == (32, 1728)


def _test_seblock():
    x = tf.random.uniform((32, 56, 56, 64))
    b = SEBlock(channels=64, reduction=16, data_format='channels_last')
    o = b(x)
    assert o.shape == (32, 56, 56, 64)
    assert get_num_params(b) == 4 * (64 * 1 * 1 + 1) + 64 * (4 * 1 * 1 + 1)


def _test_channel_shuffle():
    x = tf.convert_to_tensor(np.array([[
        [[1,5,9,13,17,21], [2,6,10,14,18,22]],
        [[3,7,11,15,19,23], [4,8,12,16,20,24]]
    ]]))
    e1 = tf.convert_to_tensor(np.array([[
        [[1,13,5,17,9,21], [2,14,6,18,10,22]],
        [[3,15,7,19,11,23], [4,16,8,20,12,24]]
    ]]))
    e2 = tf.convert_to_tensor(np.array([[
        [[1,9,17,5,13,21], [2,10,18,6,14,22]],
        [[3,11,19,7,15,23], [4,12,20,8,16,24]]
    ]]))
    o1 = ChannelShuffle(data_format='channels_last', n_groups=2)(x)
    o2 = ChannelShuffle(data_format='channels_last', n_groups=3)(x)
    assert np.all(o1.numpy() == e1.numpy())
    assert np.all(o2.numpy() == e2.numpy())


def _test_gempool():
    x = tf.random.uniform((32, 7, 7, 1024))
    m = GeMPooling(data_format='channels_last')
    o = m(x)
    assert o.shape == (32, 1, 1, 1024)
    x = tf.random.uniform((32, 1024, 7, 7))
    m = GeMPooling(data_format='channels_first')
    o = m(x)
    assert o.shape == (32, 1024, 1, 1)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    _test_Conv2d()
    _test_MaxPool2d()
    _test_Flatten()
    _test_seblock()
    _test_channel_shuffle()
    _test_gempool()

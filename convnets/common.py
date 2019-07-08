import math
import numpy as np
import tensorflow as tf



class Conv2D(tf.keras.Model):
    """Wrapper class for 2D convolution layers."""
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
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
            name='conv2d'
        ):
        super(Conv2D, self).__init__()
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
        self.kernel_initializer = kernel_initializer
        self._name = name

        if groups == 1:
            self.mode = 'regular'
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
            self.concat = tf.keras.layers.Concatenate(axis=concat_axis, name = name + '/concat')

    def call(self, x):
        if any(margin > 0 for margin in self.padding):
            if self.data_format == 'channels_first':
                paddings = [[0, 0], [0, 0], list(self.padding), list(self.padding)]
            else:
                paddings = [[0, 0], list(self.padding), list(self.padding), [0, 0]]
            x = tf.pad(x, paddings=paddings)

        if self.mode in ['regular', 'depthwise']:
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


def _test_conv2d():
    x = tf.random.uniform((32, 24, 24, 16))
    conv1 = Conv2D(in_channels=16, out_channels=40, kernel_size=3, strides=2, padding=0, dilation=1, groups=1)
    out1 = conv1(x)
    assert out1.shape == (32, 11, 11, 40)
    conv2 = Conv2D(in_channels=16, out_channels=16, kernel_size=3, strides=2, padding=0, dilation=1, groups=16)
    out2 = conv2(x)
    assert out2.shape == (32, 11, 11, 16)
    conv3 = Conv2D(in_channels=16, out_channels=40, kernel_size=3, strides=2, padding=0, dilation=1, groups=4)
    out3 = conv3(x)
    assert out3.shape == (32, 11, 11, 40)


if __name__ == '__main__':
    _test_conv2d()

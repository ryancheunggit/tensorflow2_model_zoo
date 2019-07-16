import tensorflow as tf
from common import Conv2d, MaxPool2d, get_num_params
from tensorflow.keras.layers import ReLU, Concatenate, Dropout, GlobalAveragePooling2D, Flatten

__all__ = ['SqueezeNet', 'squeezenet_v1_0', 'squeezenet_v1_1', 'squeezeresnet_v1_0', 'squeezeresnet_v1_1']


class Fire(tf.keras.Model):
    """Fire Module from SqueezeNet.

    A Fire module is comprised of:
        a squeeze convolution layer of 1x1 conv
        an expand layer with mixed 1x1 and 3x3 conv filters
    Also added option for residual connection

                             x_in  ------------------------------------------
                         (H x W x D_in )                                    |
                               |                                            |
                  1x1 Conv (1 x 1 x D_out // 8)                             |
                       (H x W x D_out // 8)                                 |
                               |                                            |
                --------------------------------                            |
                |                              |                            | x identity (optional, only applicable
    1x1 Conv (1 x 1 x D_out // 2)  3x3 Conv (3 x 3 x D_out // 2)            | when fire module does not change number
        (H x W x D_out // 2)           (H x W x D_out // 2)                 | of channels)
                |                              |                            |
                --------------------------------                            |
                                |                                           |
                           Concatenate                                      |
                         (H x W x D_out)                                    |
                                |                                           |
                                +--------------------------------------------
                              x_out
                         (H x W x D_out)
    """
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels, residual,
                 data_format='channels_last', name='fire'):
        super(Fire, self).__init__()
        self.squeeze = tf.keras.Sequential([
            Conv2d(in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1, padding=0,
                   data_format=data_format, name=name + '/squeeze_conv'),
            ReLU(name=name + '/squeeze_activ')
        ])

        self.expand1x1 = tf.keras.Sequential([
            Conv2d(in_channels=squeeze_channels, out_channels=expand1x1_channels, kernel_size=1, padding=0,
                   data_format=data_format, name=name + '/expand1x1_conv'),
            ReLU(name=name + '/expand1x1_activ')
        ])

        self.expand3x3 = tf.keras.Sequential([
            Conv2d(in_channels=squeeze_channels, out_channels=expand3x3_channels, kernel_size=3, padding=1,
                   data_format=data_format, name=name + '/expand3x3_conv'),
            ReLU(name=name + '/expand3x3_activ')
        ])

        self.concat = Concatenate(axis=-1 if data_format == 'channels_last' else 1, name=name + '/concat')
        self.residual = residual

    def call(self, x, training=False):
        if self.residual:
            identity = x

        s1 = self.squeeze(x)
        e1 = self.expand1x1(s1)
        e3 = self.expand3x3(s1)
        out = self.concat([e1, e3])

        if self.residual:
            out = out + identity
        return out


class SqueezeNet(tf.keras.Model):
    """Implementation of SqueezeNet

    reference: https://arxiv.org/abs/1602.07360.
    """
    def __init__(self, channels, residuals, init_kernel_size, init_channels, in_channels=3, input_shape=(224, 224),
                 num_classes=1000, dropout_rate=.5, data_format='channels_last', name='squeezenet'):
        super(SqueezeNet, self).__init__()
        self.features = tf.keras.Sequential(name=name + '/features')

        # init conv
        self.features.add(
            Conv2d(
                in_channels=in_channels, out_channels=init_channels, kernel_size=init_kernel_size,
                strides=2, use_bias=True, data_format=data_format, name=name + '/features/init_conv'
            )
        )
        self.features.add(ReLU(name=name + '/features/init_activ'))

        # stages of fire blocks
        in_channels = init_channels
        for i, channels_per_stage in enumerate(channels):
            self.features.add(
                MaxPool2d(
                    pool_size=3, strides=2, ceil_mode=True, data_format=data_format,
                    name=name + '/features/stage{}/pool'.format(i)
                )
            )
            for j, out_channels in enumerate(channels_per_stage):
                squeeze_channels = out_channels // 8
                expand_channels = out_channels // 2
                self.features.add(
                    Fire(
                        in_channels=in_channels, squeeze_channels=squeeze_channels,
                        expand1x1_channels=expand_channels, expand3x3_channels=expand_channels,
                        residual=(residuals is not None) and (residuals[i][j] == 1),
                        data_format=data_format, name=name + '/features/stage{}/fire{}'.format(i, j)
                    )
                )
                in_channels = out_channels

        # classifier head
        self.classifier = tf.keras.Sequential([
            Dropout(rate=dropout_rate, name=name + '/classifier/dropout'),
            Conv2d(
                in_channels=in_channels, out_channels=num_classes, kernel_size=1, data_format=data_format,
                name=name + '/classifier/final_conv'
            ),
            ReLU(name=name + '/classifier/activ'),
            GlobalAveragePooling2D(name=name + '/classifier/pool'),
            Flatten(name=name + '/classifier/flatten')
        ])

    def call(self, x, training=False):
        x = self.features(x)
        x = self.classifier(x, training=training)
        return x


def get_squeezenet(version, residual=False, **kwargs):
    assert version in ['1.0', '1.1'], 'version number {} is not supported'.format(version)

    if version == '1.0':
        channels = [[128, 128, 256], [256, 384, 384, 512], [512]]
        residuals = [[0, 1, 0], [1, 0, 1, 0], [1]]
        init_block_kernel_size = 7
        init_block_channels = 96
    elif version == '1.1':
        channels = [[128, 128], [256, 256], [384, 384, 512, 512]]
        residuals = [[0, 1], [0, 1], [0, 1, 0, 1]]
        init_block_kernel_size = 3
        init_block_channels = 64

    if not residual:
        residuals = None

    return SqueezeNet(channels=channels, residuals=residuals, init_kernel_size=init_block_kernel_size,
                      init_channels=init_block_channels, **kwargs)


def squeezenet_v1_0(**kwargs):
    return get_squeezenet(version='1.0', residual=False, **kwargs)


def squeezenet_v1_1(**kwargs):
    return get_squeezenet(version='1.1', residual=False, **kwargs)


def squeezeresnet_v1_0(**kwargs):
    return get_squeezenet(version='1.0', residual=True, **kwargs)


def squeezeresnet_v1_1(**kwargs):
    return get_squeezenet(version='1.1', residual=True, **kwargs)


def _test_fire():
    x = tf.random.uniform((32, 224, 224, 64))
    m = Fire(64, 8, 32, 32, False)
    o = m(x)
    assert o.shape == (32, 224, 224, 64) and get_num_params(m) == (64 + 1) * 8 + (8 + 1) * 32 + (8 * 9 + 1) * 32

    m2 = Conv2d(64, 64, 3, 1, 1)
    o = m2(x)
    assert o.shape == (32, 224, 224, 64) and get_num_params(m2) == (64 * 9 + 1) * 64
    # compare to regular 3x3 conv, the fire block is 10 times smaller in number of parameters


def _test_squeezenet():
    x = tf.random.uniform((32, 224, 224, 3))

    model = squeezenet_v1_0()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 1248424

    model = squeezenet_v1_1()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 1235496

    model = squeezeresnet_v1_0()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 1248424

    model = squeezeresnet_v1_1()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 1235496


if __name__ == '__main__':
    _test_fire()
    _test_squeezenet()

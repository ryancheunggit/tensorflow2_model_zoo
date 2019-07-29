import tensorflow as tf
from activation import ReLU6
from common import Conv2d, get_num_params
from tensorflow.keras.layers import BatchNormalization, Flatten, AveragePooling2D


__all__ = ['MobileNetV2']


class InvertedResidual(tf.keras.Model):
    def __init__(self, in_channels, out_channels, strides, expansion_ratio=1, data_format='channels_last',
                 name='inverted_residual'):
        super(InvertedResidual, self).__init__(name=name)
        self.residual_connection = (in_channels == out_channels) and (strides == 1)
        inner_channels = in_channels * expansion_ratio

        self.block = tf.keras.Sequential()

        # expansion
        if expansion_ratio != 1:
            self.block.add(
                Conv2d(in_channels, inner_channels, kernel_size=1, strides=1, padding=0, use_bias=False,
                       data_format=data_format, name=name + '/expansion')
            )
            self.block.add(BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + 'bn0'))
            self.block.add(ReLU6(name=name + 'activ0'))

        # depthwise conv
        self.block.add(
            Conv2d(inner_channels, inner_channels, kernel_size=3, strides=strides, padding=1,
                   groups=inner_channels, use_bias=False, data_format=data_format, name=name + '/depthwise_conv')
        )
        self.block.add(BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/bn1'))
        self.block.add(ReLU6(name=name + '/activ1'))

        # point-wise conv
        self.block.add(
            Conv2d(inner_channels, out_channels, kernel_size=1, strides=1, padding=0, use_bias=False,
                   data_format=data_format, name=name + '/pointwise_conv')
        )
        self.block.add(BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/bn2'))

    def call(self, x, training=False):
        if self.residual_connection:
            identity = x
            return self.block(x, training=training) + identity
        else:
            return self.block(x, training=training)


INIT_CHANNELS = 32
OUT_CHANNELS = 1280
T = [1, 6, 6, 6, 6, 6, 6]
C = [16, 24, 32, 64, 96, 160, 320]
N = [1, 2, 3, 4, 3, 3, 1]
S = [1, 2, 2, 2, 1, 2, 1]


class MobileNetV2(tf.keras.Model):
    """Implementation of MobileNetV2.

    reference: https://arxiv.org/abs/1801.04381
    """
    def __init__(self, in_channels=3, num_classes=1000, init_channels=INIT_CHANNELS, out_channels=OUT_CHANNELS,
                 t=T, c=C, n=N, s=S, width_multiplier=1., data_format='channels_last', name='MobileNetV2'):
        super(MobileNetV2, self).__init__(name=name)
        init_channels = int(init_channels * width_multiplier)
        out_channels = int(out_channels * width_multiplier) if width_multiplier > 1 else out_channels
        self.features = tf.keras.Sequential()

        init_conv = tf.keras.Sequential([
            Conv2d(in_channels, init_channels, kernel_size=3, strides=2, padding=1, use_bias=False,
                   data_format=data_format, name=name + '/init/conv0'),
            BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/init/bn0'),
            ReLU6(name=name + '/init/activ0')
        ])

        residual_bottlenecks = tf.keras.Sequential()
        input_channels = init_channels
        for nt, nc, nn, ns in zip(t, c, n, s):
            output_channels = int(nc * width_multiplier)
            for i in range(nn):
                if i == 0:
                    residual_bottlenecks.add(InvertedResidual(input_channels, output_channels, ns, expansion_ratio=nt))
                else:
                    residual_bottlenecks.add(InvertedResidual(input_channels, output_channels, 1, expansion_ratio=nt))
                input_channels = output_channels

        final_conv = tf.keras.Sequential([
            Conv2d(input_channels, out_channels, kernel_size=1, strides=1, padding=0, use_bias=False,
                   data_format=data_format, name=name + '/final_conv'),
            BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/final_bn'),
            ReLU6(name=name + '/final_activ')
        ])

        self.features.add(init_conv)
        self.features.add(residual_bottlenecks)
        self.features.add(final_conv)
        self.features.add(AveragePooling2D(pool_size=7, data_format=data_format, name='global_avg_pool'))

        self.classifier = tf.keras.Sequential([
            Conv2d(out_channels, num_classes, kernel_size=1, strides=1, padding=0, use_bias=True,
                   data_format=data_format, name='classifier'),
            Flatten()
        ])

    def call(self, x, training=False):
        features = self.features(x, training=training)
        out = self.classifier(features, training=training)
        return out


def _test_mobilenet():
    x = tf.random.uniform((32, 224, 224, 3))
    m = MobileNetV2()
    o = m(x)
    assert o.shape == (32, 1000)
    assert get_num_params(m) == 3504872


if __name__ == '__main__':
    _test_mobilenet()

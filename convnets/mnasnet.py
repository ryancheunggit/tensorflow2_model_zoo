import tensorflow as tf
from common import Conv2d, SEBlock, get_num_params
from tensorflow.keras.layers import BatchNormalization, ReLU, AveragePooling2D, Flatten, Dense


__all__ = ['MnasNet']


class SepConv(tf.keras.Model):
    def __init__(self, in_channels, out_channels, data_format='channels_last', name='sepconv_block'):
        super(SepConv, self).__init__(name=name)
        channel_axis = -1 if data_format == 'channels_last' else 1
        self.dwconv3x3 = tf.keras.Sequential([
            Conv2d(in_channels, in_channels, kernel_size=3, strides=1, padding=1, groups=in_channels, use_bias=False,
                   data_format=data_format, name=name + '/dwconv3x3'),
            BatchNormalization(axis=channel_axis, name=name + '/bn'),
            ReLU(name=name + '/activ')
        ])
        self.conv1x1 = tf.keras.Sequential([
            Conv2d(in_channels, out_channels, kernel_size=1, strides=1, padding=0, use_bias=False,
                   data_format=data_format, name=name + '/conv1x1'),
            BatchNormalization(axis=channel_axis, name=name + '/bn')
        ])

    def call(self, x, training=False):
        x = self.dwconv3x3(x, training=training)
        x = self.conv1x1(x, training=training)
        return x


class MBConv(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size, strides, expansion_factor, se_ratio=None,
                 data_format='channels_last', name='mbconv_block'):
        assert kernel_size in (3, 5)
        super(MBConv, self).__init__(name=name)
        self.residual = (in_channels == out_channels) and (strides == 1)
        inner_channels = in_channels * expansion_factor
        channel_axis = -1 if data_format == 'channels_last' else 1
        dwconv_padding = 1 if kernel_size == 3 else 2
        self.conv1x1_expand = tf.keras.Sequential([
            Conv2d(in_channels, inner_channels, kernel_size=1, strides=1, padding=0, use_bias=False,
                   data_format=data_format, name=name + '/expand/conv1x1'),
            BatchNormalization(axis=channel_axis, name=name + '/expand/bn'),
            ReLU(name=name + '/expand/activ')
        ])

        self.dwconv = tf.keras.Sequential([
            Conv2d(inner_channels, inner_channels, kernel_size=kernel_size, strides=strides, padding=dwconv_padding,
                   groups=inner_channels, use_bias=False, data_format=data_format, name=name + '/dwconv/dwconv'),
            BatchNormalization(axis=channel_axis, name=name + '/dwconv/bn'),
            ReLU(name=name + '/dwconv/activ')
        ])
        se_reduction = int(inner_channels * se_ratio) if se_ratio else None
        self.se = SEBlock(inner_channels, se_reduction, data_format, name=name + '/se') if se_ratio else None

        self.conv1x1_project = tf.keras.Sequential([
            Conv2d(inner_channels, out_channels, kernel_size=1, strides=1, padding=0, use_bias=False,
                   data_format=data_format, name=name + '/project/conv1x1'),
            BatchNormalization(axis=channel_axis, name=name + '/project/bn')
        ])

    def call(self, x, training=False):
        if self.residual:
            identity = x

        x = self.conv1x1_expand(x, training=training)
        x = self.dwconv(x, training=training)
        if self.se:
            x = self.se(x, training=training)
        x = self.conv1x1_project(x, training=training)

        if self.residual:
            x = x + identity
        return x


INIT_CHANNELS = 32
OUT_CHANNELS = 1280
N = [   1,    2,    3,    4,    2,    3,    1]  # number of blocks per stage
C = [  16,   24,   40,   80,  112,  160,  320]  # number of channels per stage
S = [   1,    2,    2,    2,    1,    2,    1]  # strides for downsample per stage
R = [None, None,  .25, None,  .25,  .25, None]  # se ratio per stage
K = [   3,    3,    5,    3,    3,    5,    3]  # depthwise conv kernel size
E = [None,    6,    3,    6,    6,    6,    6]  # expansion factor per stage


def _round_to_multiple_of(channels, divisor, bias=.9):
    new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < bias * divisor:
        new_channels += divisor
    return new_channels


def _scale_depth(stage_channels, width_scale):
    return [_round_to_multiple_of(channels * width_scale, 8) for channels in stage_channels]


class MnasNet(tf.keras.Model):
    """Implementation of MnasNet.

    reference: https://arxiv.org/abs/1807.11626
    """
    def __init__(self, in_channels=3, num_classes=1000, init_channels=INIT_CHANNELS, out_channels=OUT_CHANNELS,
                 n=N, c=C, s=S, r=R, k=K, e=E, depth_scale=1, data_format='channels_last', name='MnasNet'):
        super(MnasNet, self).__init__(name=name)
        channel_axis = -1 if data_format == 'channels_last' else 1
        if depth_scale != 1:
            c = _scale_depth(c, depth_scale)

        self.init_conv = tf.keras.Sequential([
            Conv2d(in_channels, init_channels, kernel_size=3, strides=2, padding=1, use_bias=False,
                   data_format=data_format, name=name + '/init/conv0'),
            BatchNormalization(axis=channel_axis, name=name + '/init/bn'),
            ReLU(name=name + '/init/activ')
        ])

        self.features = tf.keras.Sequential(name='features')
        input_channels = init_channels
        for i, (num_layers, channels, strides, se_r, kernel_size, expand_factor) in enumerate(zip(n, c, s, r, k, e)):
            if i == 0:
                stage = SepConv(init_channels, channels, data_format=data_format, name=name + '/features/stage0')
                input_channels = channels
            else:
                stage = tf.keras.Sequential(name=name + '/features/stage{}'.format(i))
                for j in range(num_layers):
                    stage.add(MBConv(
                            in_channels=input_channels,
                            out_channels=channels,
                            kernel_size=kernel_size,
                            strides=strides if j == 0 else 1,
                            expansion_factor=expand_factor,
                            se_ratio=se_r,
                            data_format=data_format,
                            name=name + '/features/stage{}/unit{}'.format(i, j)
                        )
                    )
                    input_channels = channels
            self.features.add(stage)
        final_conv = tf.keras.Sequential([
            Conv2d(input_channels, out_channels, kernel_size=1, strides=1, padding=0, use_bias=False,
                   data_format=data_format, name=name + '/final_conv'),
            BatchNormalization(axis=channel_axis, name=name + '/final_bn'),
            ReLU(name=name + '/final_activ')
        ])
        self.features.add(final_conv)
        self.features.add(AveragePooling2D(pool_size=7, data_format=data_format, name='global_avg_pool'))
        self.classifier = tf.keras.Sequential([
            Flatten(),
            Dense(units=num_classes)
        ])


    def call(self, x, training=False):
        x = self.init_conv(x)
        x = self.features(x)
        x = self.classifier(x)
        return x



def _test_sepconv_block():
    x = tf.random.uniform((32, 112, 112, 32))
    m = SepConv(32, 16)
    o = m(x)
    assert o.shape == (32, 112, 112, 16)
    assert get_num_params(m) == 32 * 1 * (3 * 3 + 0) + 32 * 2 + 32 * 16 * (1 * 1 + 0) + 16 * 2


def _test_mbconvblock():
    x = tf.random.uniform((32, 112, 112, 16))
    m = MBConv(in_channels=16, out_channels=24, kernel_size=3, strides=2, expansion_factor=6)
    x = m(x)
    assert x.shape == (32, 56, 56, 24)
    conv1_expand_params = 16 * 16 * 6 * (1 * 1 + 0) + 16 * 6 * 2
    conv3_params = 1 * 16 * 6 * (3 * 3 + 0) + 16 * 6 * 2
    conv1_projec_params = 16 * 6 * 24 * (1 * 1 + 0) + 24 * 2
    assert get_num_params(m) == conv1_expand_params + conv3_params + conv1_projec_params
    m2 = MBConv(in_channels=24, out_channels=24, kernel_size=3, strides=1, expansion_factor=6)
    x = m2(x)
    assert x.shape == (32, 56, 56, 24)
    assert m2.residual
    conv1_expand_params = 24 * 24 * 6 * (1 * 1 + 0) + 24 * 6 * 2
    conv3_params = 1 * 24 * 6 * (3 * 3 + 0) + 24 * 6 * 2
    conv1_projec_params = 24 * 6 * 24 * (1 * 1 + 0) + 24 * 2
    assert get_num_params(m2) == conv1_expand_params + conv3_params + conv1_projec_params
    m3 = MBConv(in_channels=24, out_channels=40, kernel_size=5, strides=2, expansion_factor=3, se_ratio=.25)
    x = m3(x)
    assert x.shape == (32, 28, 28, 40)
    assert get_num_params(m3) == 7428
    m4 = MBConv(in_channels=40, out_channels=40, kernel_size=5, strides=1, expansion_factor=3, se_ratio=.25)
    x = m4(x)
    assert x.shape == (32, 28, 28, 40)
    assert m4.residual


def get_mnasnet_a1_s1():
    return MnasNet()


def get_mnasnet_b1_s1():
    INIT_CHANNELS = 32
    OUT_CHANNELS = 1280
    N = [   1,    3,    3,    3,    2,    4,    1]  # number of blocks per stage
    C = [  16,   24,   40,   80,   96,  192,  320]  # number of channels per stage
    S = [   1,    2,    2,    2,    1,    2,    1]  # strides for downsample per stage
    R = [None, None, None, None, None, None, None]  # se ratio per stage
    K = [   3,    3,    5,    5,    3,    5,    3]  # depthwise conv kernel size
    E = [None,    3,    3,    6,    6,    6,    6]  # expansion factor per stage
    return MnasNet(n=N, c=C, s=S, r=R, k=K, e=E)



def _test_mnasnet():
    x = tf.random.uniform((32, 224, 224, 3))
    m = get_mnasnet_a1_s1()
    o = m(x)
    assert o.shape == (32, 1000)
    assert get_num_params(m) == 3665608
    m2 = get_mnasnet_b1_s1()
    o2 = m2(x)
    assert o2.shape == (32, 1000)
    assert get_mnasnet_b1_s1(m2) == 4383312


if __name__ == '__main__':
    _test_sepconv_block()
    _test_mbconvblock()
    _test_mnasnet()

import tensorflow as tf
from common import Conv2d, MaxPool2d, SEBlock, ChannelShuffle, get_num_params
from tensorflow.keras.layers import ReLU, Concatenate, BatchNormalization, AveragePooling2D, Flatten


__all__ = ['ShuffleNetV2', 'get_shufflenetv2']


class DWConv3x3_Block(tf.keras.Model):
    def __init__(self, channels, strides, activ=False, data_format='channels_last', name='DWConv3x3'):
        super(DWConv3x3_Block, self).__init__(name=name)
        self.conv = Conv2d(channels, channels, kernel_size=3, strides=strides, padding=1, groups=channels,
                           use_bias=False, data_format=data_format, name=name + '/conv2d')
        self.bn = BatchNormalization(axis=-1 if data_format=='channels_last' else 1, name=name + '/bn')
        self.activ = ReLU(name=name + '/activ') if activ else None

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        if self.activ:
            x = self.activ(x)
        return x


class Conv1x1_Block(tf.keras.Model):
    def __init__(self, in_channels, out_channels, activ=True, data_format='channels_last', name='Conv1x1'):
        super(Conv1x1_Block, self).__init__(name=name)
        self.conv = Conv2d(in_channels, out_channels, kernel_size=1, use_bias=False, data_format=data_format,
                           name=name + '/conv2d')
        self.bn = BatchNormalization(axis=-1 if data_format=='channels_last' else 1, name=name + '/bn')
        self.activ = ReLU(name=name + '/activ') if activ else None

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        if self.activ:
            x = self.activ(x)
        return x


class ShuffleNetV2BasicUnit(tf.keras.Model):
    """ShuffleNetV2 basic unit.

           x_in
             |
    ---ChannelSplit---
    |                |
    x1              x2--------------
    |                |             |
    |             1x1 Conv         |
    |                | BN ReLU     |
    |             3x3 DWConv       |
    |                | BN          | x2 identity
    |             1x1 Conv         |
    |                | BN (ReLU*)  |
    |             [SE Block]       |
    |                +-------------- [Residual Connection] (ReLU*)
    |                |
    ------Concat------
             |
       ChannelShuffle
             |
           x_out

    after channelsplit, x1, x2 each contains half the total channels from x_in.
    if use residual connection, activation for the second 1x1Conv will be moved after addition.
    """

    def __init__(self, channels, n_groups=2, se_r=None, residual=False, data_format='channels_last',
                 name='shufflenetv2_basic_unit'):
        assert channels % 2 == 0
        super(ShuffleNetV2BasicUnit, self).__init__(name=name)
        self.channel_axis = -1 if data_format == 'channels_last' else 1

        mid_channels = channels // 2
        self.branch = tf.keras.Sequential([
            Conv1x1_Block(mid_channels, mid_channels, activ=True, data_format=data_format, name=name + '/conv1x1_0'),
            DWConv3x3_Block(mid_channels, strides=1, activ=False, data_format=data_format, name=name + '/dwconv3x3'),
            Conv1x1_Block(mid_channels, mid_channels, activ=residual == False, data_format=data_format,
                          name=name + '/conv1x1_1')
        ])

        self.se = SEBlock(mid_channels, se_r, data_format=data_format, name=name + '/se') if se_r else None
        self.residual_activ = ReLU(name=name + '/residual_activ') if residual else None

        self.concat = Concatenate(axis=self.channel_axis, name=name + '/concat')
        self.shuffle = ChannelShuffle(data_format, n_groups, name=name + '/channel_shuffle')

    def call(self, x, training=False):
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=self.channel_axis)
        if self.residual_activ:
            x2_identity = x2

        x2 = self.branch(x2, training=training)
        if self.se:
            x2 = self.se(x2, training=training)

        if self.residual_activ:
            x2 = x2 + x2_identity
            x2 = self.residual_activ(x2)

        concat = self.concat([x1, x2])
        x = self.shuffle(concat)
        return x


class ShuffleNetV2DownsampleUnit(tf.keras.Model):
    """ShuffleNetV2 downsample unit.

    Downsample unit
                x_in
                  |
        -------------------
        |                 |
        x1                x2
        |                 |
     3x3 DWConv        1x1 Conv
      stride=2            | BN ReLU
        | BN           3x3 DWConv
     1x1 Conv           stride=2
        | BN ReLU         | BN
        |              1x1 Conv
        |                 | BN ReLU
        |              [SE block]
        |                 |
        ------Concat-------
                 |
          ChannelShuffle
                 |
               x_out

    x1, x2 are the same as x_in.
    activation after the second relu can be delayed in case of use residual.
    """
    def __init__(self, in_channels, out_channels, n_groups=2, se_r=None, data_format='channels_last',
                 name='shufflenetv2_downsample_unit'):
        super(ShuffleNetV2DownsampleUnit, self).__init__(name=name)
        self.channel_axis = -1 if data_format == 'channels_last' else 1
        branch_out_channels = out_channels - in_channels
        branch_mid_channels = out_channels // 2
        assert branch_mid_channels % 2 == 0

        self.short_cut_branch = tf.keras.Sequential([
            DWConv3x3_Block(in_channels, strides=2, activ=False, data_format=data_format,
                            name=name + '/shortcut_dwconv3x3'),
            Conv1x1_Block(in_channels, in_channels, activ=True, data_format=data_format,
                          name=name + '/shortcut_conv1x1')
        ])
        self.branch = tf.keras.Sequential([
            Conv1x1_Block(in_channels, branch_mid_channels, activ=True, data_format=data_format,
                          name=name + '/branch_conv1x1_0'),
            DWConv3x3_Block(branch_mid_channels, strides=2, activ=False, data_format=data_format,
                            name=name + '/branch_dwconv3x3'),
            Conv1x1_Block(branch_mid_channels, branch_out_channels, activ=True, data_format=data_format,
                          name=name + '/branch_conv1x1_1')
        ])
        self.se = SEBlock(branch_out_channels, se_r, data_format=data_format, name=name + '/se') if se_r else None

        self.concat = Concatenate(axis=self.channel_axis, name=name + '/concat')
        self.shuffle = ChannelShuffle(data_format, n_groups, name=name + '/channel_shuffle')


    def call(self, x, training=False):
        x1, x2 = x, x
        x1 = self.short_cut_branch(x1, training=training)
        x2 = self.branch(x2, training=training)
        if self.se:
            x2 = self.se(x2, training=training)

        concat = self.concat([x1, x2])
        x = self.shuffle(concat)
        return x


INIT_CHANNELS = 24
OUT_CHANNELS = 1024
C = [116, 232, 464]
N = [4, 8, 4]


class ShuffleNetV2(tf.keras.Model):
    """Implementation of ShuffleNetV2.

    reference: https://arxiv.org/abs/1807.11164
    """
    def __init__(self, in_channels=3, num_classes=1000, n_groups=2, se_r=None, residual=False,
                 init_channels=INIT_CHANNELS, out_channels=OUT_CHANNELS, c=C, n=N, data_format='channels_last',
                 name='ShuffleNetV2'):
        super(ShuffleNetV2, self).__init__(name=name)
        self.init_conv = tf.keras.Sequential([
            Conv2d(in_channels, init_channels, kernel_size=3, strides=2, padding=1, use_bias=False,
                   data_format=data_format, name=name + '/init/conv0'),
            BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/init/bn0'),
            ReLU(name=name + '/init/activ0'),
            MaxPool2d(pool_size=3, strides=2, padding=1, data_format=data_format, name=name + '/init/pool0')
        ])

        self.features = tf.keras.Sequential()
        input_channels = init_channels
        for i, (nn, nc) in enumerate(zip(n, c)):
            stage = tf.keras.Sequential(name=name + '/stage{}'.format(i))
            for j in range(nn):
                if j == 0:
                    stage.add(ShuffleNetV2DownsampleUnit(input_channels, nc, n_groups, se_r, data_format))
                else:
                    stage.add(ShuffleNetV2BasicUnit(nc, n_groups, se_r, residual, data_format))
            input_channels = nc
            self.features.add(stage)
        final_conv = tf.keras.Sequential([
            Conv2d(input_channels, out_channels, kernel_size=1, strides=1, padding=0, use_bias=False,
                   data_format=data_format, name=name + '/final_conv'),
            BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/final_bn'),
            ReLU(name=name + '/final_activ')
        ])
        self.features.add(final_conv)
        self.features.add(AveragePooling2D(pool_size=7, data_format=data_format, name='global_avg_pool'))
        self.classifier = tf.keras.Sequential([
            Conv2d(out_channels, num_classes, kernel_size=1, strides=1, padding=0, use_bias=True,
                   data_format=data_format, name='classifier'),
            Flatten()
        ])

    def call(self, x, training=False):
        x = self.init_conv(x, training=training)
        x = self.features(x, training=training)
        x = self.classifier(x)
        return x


def get_shufflenetv2(in_channels=3, num_classes=1000, n_groups=2, se_r=None, residual=False, width_scale='1',
                     data_format='channels_last'):
    name = 'shufflenetv2'
    if residual:
        name = 'res_' + name
    if se_r:
        name = 'se_' + name
    name = name + '_w' + width_scale

    if width_scale == '1':
        c = [116, 232, 464]
    elif width_scale == '.5':
        c = [48, 96, 192]
    elif width_scale == '1.5':
        c = [176, 352, 704]
    elif width_scale == '2.0':
        c = [244, 488, 976]
    out_channels = 2048 if width_scale == '2.0' else 1024
    return ShuffleNetV2(in_channels, num_classes, n_groups, se_r, residual, out_channels=out_channels, c=c,
                        data_format=data_format, name=name)

def _test_basic_unit():
    x = tf.random.uniform((32, 28, 28, 116))
    m1 = ShuffleNetV2BasicUnit(116)
    o1 = m1(x)
    m2 = ShuffleNetV2BasicUnit(116, se_r=16)
    o2 = m2(x)
    m3 = ShuffleNetV2BasicUnit(116, residual=True)
    o3 = m3(x)
    m4 = ShuffleNetV2BasicUnit(116, se_r=16, residual=True)
    o4 = m4(x)
    for i, (o, m) in enumerate(zip([o1, o2, o3, o4], [m1, m2, m3, m4])):
        assert o.shape == (32, 28, 28, 116)
        if i % 2 == 0:
            assert get_num_params(m) == 7598
        else:
            assert get_num_params(m) == 8007


def _test_downsample_unit():
    x = tf.random.uniform((32, 56, 56, 24))
    m1 = ShuffleNetV2DownsampleUnit(24, 116, se_r=None)
    o1 = m1(x)
    m2 = ShuffleNetV2DownsampleUnit(24, 116, se_r=16)
    o2 = m2(x)
    assert o1.shape == o2.shape == (32, 28, 28, 116)
    assert get_num_params(m1) == 8554
    assert get_num_params(m2) == 9571


def _test_shufflenetv2():
    x = tf.random.uniform((32, 224, 224, 3))
    m1 = ShuffleNetV2()
    o1 = m1(x)
    m2 = ShuffleNetV2(se_r=16)
    o2 = m2(x)
    m3 = get_shufflenetv2(residual=True)
    o3 = m3(x)
    m4 = get_shufflenetv2(se_r=16, residual=True)
    o4 = m4(x)
    m5 = ShuffleNetV2(out_channels=2048, c=[244, 488, 976])
    o5 = m5(x)
    m6 = get_shufflenetv2(se_r=16, residual=True, n_groups=4, width_scale='2.0')
    o6 = m6(x)
    assert o1.shape == o2.shape == o3.shape == o4.shape == o5.shape == o6.shape == (32, 1000)
    assert get_num_params(m1) == get_num_params(m3) == 2279760
    assert get_num_params(m2) == get_num_params(m4) == 2322948
    assert get_num_params(m5) == 7403600
    assert get_num_params(m6) == 7594888


if __name__ == '__main__':
    _test_basic_unit()
    _test_downsample_unit()
    _test_shufflenetv2()

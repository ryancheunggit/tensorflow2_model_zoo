import tensorflow as tf
from common import Conv2d, MaxPool2d, SEBlock, get_num_params
from tensorflow.keras.layers import BatchNormalization, ReLU, GlobalAveragePooling2D, Dense


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d',
    'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'seresnext_50', 'seresnext_101']


def conv3x3(in_channels, out_channels, strides=1, dilation=1, groups=1, data_format='channels_last', name='conv3x3'):
    return Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=3, strides=strides, padding=dilation,
        dilation=dilation, groups=groups, use_bias=False, data_format=data_format, name=name)


def conv1x1(in_channels, out_channels, strides=1, data_format='channels_last', name='conv1x1'):
    return Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=1, strides=strides, use_bias=False,
        data_format=data_format, name=name)


class ResBlock(tf.keras.Model):
    """Basic Resudial Block.


    x_in (64-d) ------
         |           |
    Conv 3x3x64 BN   |
         | relu      | x identity (downsample via conv1x1 if needed)
    Conv 3x3x64      |
         |           |
     [SEBlock]       |
         +------------
         | relu
    x_out (64-d)
    """
    expansion = 1

    def __init__(self, in_channels, channels, strides=1, downsample=None, groups=1, base_width=64, dilation=1,
                 se_reduction=None, data_format='channels_last', name='residual_block'):
        super(ResBlock, self).__init__()
        assert (groups == 1) and (base_width == 64) and (dilation == 1)
        self.conv1 = conv3x3(in_channels, channels, strides, data_format=data_format, name=name + '/conv1')
        self.bn1 = BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/bn1')
        self.conv2 = conv3x3(channels, channels, data_format=data_format, name=name + '/conv2')
        self.bn2 = BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/bn2')
        self.downsample = downsample
        self.activation = ReLU()
        self.se = None
        if se_reduction:
            self.se = SEBlock(channels=channels * self.expansion, reduction=se_reduction, data_format=data_format,
                              name=name + '/se')

    def call(self, x, training=False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample:
            identity = self.downsample(identity)

        if self.se:
            out = self.se(out)

        out = out + identity
        out = self.activation(out)
        return out


class ResBottleneckBlock(tf.keras.Model):
    """Residual Block with bottlenet.

    x_in (64-d) -------
         |            |
    Conv 1x1 BN       |
         | relu       |
    Conv 3x3 BN       | x identity (downsample via conv1x1 if needed)
         | relu       |
    Conv 1x1 BN       |
         |            |
     [seblock]        |
         |            |
         +-------------
         | relu
    x_out (64-d)
    """
    expansion = 4

    def __init__(self, in_channels, channels, strides=1, downsample=None, groups=1, base_width=64, dilation=1,
                 se_reduction=None, data_format='channels_last', name='residual_bottleneck_block'):
        super(ResBottleneckBlock, self).__init__()
        mid_channels = int(channels * (base_width / 64.)) * groups

        self.conv1 = conv1x1(in_channels, mid_channels, name=name + '/conv1')
        self.bn1 = BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/bn1')

        self.conv2 = conv3x3(mid_channels, mid_channels, strides, dilation, groups, name='/conv2')
        self.bn2 = BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/bn2')

        self.conv3 = conv1x1(mid_channels, channels * self.expansion, name=name + '/conv3')
        self.bn3 = BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/bn3')

        self.activation = ReLU()
        self.downsample = downsample
        self.se = None
        if se_reduction:
            self.se = SEBlock(channels=channels * self.expansion, reduction=se_reduction, data_format=data_format,
                              name=name + '/se')

    def call(self, x, training=False):
        identity = x

        # reduction
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.activation(out)

        # expansion
        out = self.conv3(out)
        out = self.bn3(out, training=training)

        if self.downsample:
            identity = self.downsample(identity)

        if self.se:
            out = self.se(out)

        out = out + identity
        out = self.activation(out)
        return out


class ResNet(tf.keras.Model):
    """Implementation of ResNet, ResNeXt, Wide_ResNet, and SEResNeXt

    ResNet: https://arxiv.org/abs/1512.03385
    ResNeXt: https://arxiv.org/abs/1611.05431
    WideResNet: https://arxiv.org/abs/1605.07146
    SEModule: https://arxiv.org/abs/1709.01507

    Argument:
        groups: the cardinality parameter for NeXt part
        width_per_group: the D parameter for NeXt part
        se_reduction: the r parameter for SE part
    """
    def __init__(self, res_block, layers, in_channels, num_classes=1000, groups=1, width_per_group=64,
                 dilation_for_stride=None, se_reduction=0, data_format='channels_last', name='resnet'):
        super(ResNet, self).__init__()
        self.channels = 64
        self.dilation = 1
        if dilation_for_stride is None:
            dilation_for_stride = [False, False, False]
        self.se_reduction = se_reduction
        self.groups = groups
        self.base_width = width_per_group
        self.stage0 = tf.keras.Sequential([
            Conv2d(in_channels, self.channels, kernel_size=7, strides=2, padding=3, use_bias=False,
                   name=name + '/stage0/conv1'),
            BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/stage0/bn1'),
            ReLU(name=name + '/stage0/relu'),
            MaxPool2d(pool_size=3, strides=2, padding=1, data_format=data_format, name=name + '/stage0/maxpool')
        ])
        self.stage1 = self._make_layer(block=res_block, channels=64, blocks=layers[0], se_reduction=se_reduction)
        self.stage2 = self._make_layer(block=res_block, channels=128, blocks=layers[1], strides=2,
                                       dilation=dilation_for_stride[0], se_reduction=se_reduction)
        self.stage3 = self._make_layer(block=res_block, channels=256, blocks=layers[2], strides=2,
                                       dilation=dilation_for_stride[1], se_reduction=se_reduction)
        self.stage4 = self._make_layer(block=res_block, channels=512, blocks=layers[3], strides=2,
                                       dilation=dilation_for_stride[2], se_reduction=se_reduction)
        self.globalavgpool = GlobalAveragePooling2D(data_format=data_format, name='features')
        self.fc = Dense(units=num_classes, name='last_linear')

    def _make_layer(self, block, channels, blocks, strides=1, dilation=False, se_reduction=0,
                    data_format='channels_last', name='resnet_layer'):
        downsample = None
        previous_dilation = self.dilation

        if dilation:
            self.dilation *= strides
            strides = 1
        if strides != 1 or self.channels != channels * block.expansion:
            downsample = tf.keras.Sequential([
                conv1x1(self.channels, channels * block.expansion, strides, name=name + '/identity_downsample'),
                BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/identity_down_bn')
            ])

        layers = []
        layers.append(block(self.channels, channels, strides, downsample, self.groups, self.base_width,
                            dilation=previous_dilation, se_reduction=se_reduction, data_format=data_format,
                            name=name + '/block_1'))
        for i in range(1, blocks):
            layers.append(block(self.channels, channels, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, se_reduction=se_reduction, data_format=data_format,
                                name=name + '/block_{}'.format(i)))
        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        x = self.stage0(x, training=training)
        x = self.stage1(x, training=training)
        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.globalavgpool(x)
        x = self.fc(x)
        return x


def _test_resblock():
    x = tf.random.uniform((32, 112, 112, 64))
    block = ResBlock(in_channels=64, channels=64, strides=1)
    out = block(x)
    assert out.shape == (32, 112, 112, 64)
    block = ResBlock(in_channels=64, channels=64, strides=2, downsample=conv1x1(64, 64, 2))
    out = block(x)
    assert out.shape == (32, 56, 56, 64)


def _test_resbottlenetblock():
    x = tf.random.uniform((32, 112, 112, 64))
    block = ResBottleneckBlock(in_channels=64, channels=64, strides=1, downsample=conv1x1(64, 256, 1))
    out = block(x)
    assert out.shape == (32, 112, 112, 256)


def resnet18(in_channels=3, num_classes=1000):
    return ResNet(ResBlock, [2, 2, 2, 2], in_channels, num_classes=num_classes)


def resnet34(in_channels=3, num_classes=1000):
    return ResNet(ResBlock, [3, 4, 6, 3], in_channels)


def resnet50(in_channels=3, num_classes=1000):
    return ResNet(ResBottleneckBlock, [3, 4, 6, 3], in_channels, num_classes=num_classes)


def resnet101(in_channels=3, num_classes=1000):
    return ResNet(ResBottleneckBlock, [3, 4, 23, 3], in_channels, num_classes=num_classes)


def resnet152(in_channels=3, num_classes=1000):
    return ResNet(ResBottleneckBlock, [3, 8, 36, 3], in_channels, num_classes=num_classes)


def resnext50_32x4d(in_channels=3, num_classes=1000):
    return ResNet(ResBottleneckBlock, [3, 4, 6, 3], in_channels, num_classes=num_classes, groups=32, width_per_group=4)


def resnext101_32x8d(in_channels=3, num_classes=1000):
    return ResNet(ResBottleneckBlock, [3, 4, 23, 3], in_channels, num_classes=num_classes, groups=32, width_per_group=4)


def wide_resnet50_2(in_channels=3, num_classes=1000):
    return ResNet(ResBottleneckBlock, [3, 4, 6, 3], in_channels, num_classes=num_classes, width_per_group=64 * 2)


def wide_resnet101_2(in_channels=3, num_classes=1000):
    return ResNet(ResBottleneckBlock, [3, 4, 23, 3], in_channels, num_classes=num_classes, width_per_group=64 * 2)


def seresnext_50(in_channels=3, num_classes=1000):
    return ResNet(ResBottleneckBlock, [3, 4, 6, 3], in_channels, num_classes=num_classes, groups=32, width_per_group=4,
                  se_reduction=16)


def seresnext_101(in_channels=3, num_classes=1000):
    return ResNet(ResBottleneckBlock, [3, 4, 23, 3], in_channels, num_classes=num_classes, groups=32, width_per_group=4,
                  se_reduction=16)


def _test_resnet():
    x = tf.random.uniform((32, 224, 224, 3))
    model = resnet18()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 11689512

    model = resnet34()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 21797672

    model = resnet50()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 25557032

    model = resnet101()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 44549160

    model = resnet152()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 60192808

    model = resnext50_32x4d()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 25028904

    model = resnext101_32x8d()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 44177704

    model = wide_resnet50_2()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 68883240

    model = wide_resnet101_2()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 126886696

    model = seresnext_50()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 27559896

    model = seresnext_101()
    out = model(x)
    assert out.shape == (32, 1000) and get_num_params(model) == 48955416


if __name__ == '__main__':
    _test_resblock()
    _test_resbottlenetblock()
    _test_resnet()

import tensorflow as tf
from common import Conv2d, get_num_params, MaxPool2d
from tensorflow.keras.layers import (
    BatchNormalization, ReLU, Dropout, Concatenate, AveragePooling2D, GlobalAveragePooling2D, Dense
)


__all__ = ['DenseNet']


class PreActConv2D(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding,
                 data_format='channels_last', name='pre_act_conv_block', return_preact=False):
        super(PreActConv2D, self).__init__(name=name)
        self.batchnorm = BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/bn')
        self.activ = ReLU(name=name + '/activ')
        self.conv2d = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            data_format=data_format,
            name=name + '/conv2d'
        )
        self.return_preact=return_preact

    def call(self, x, training=False):
        x = self.batchnorm(x, training=training)
        preact = self.activ(x)
        x = self.conv2d(preact)
        if self.return_preact:
            return x, preact
        else:
            return x


def preact_conv1x1(in_channels, out_channels, strides=1, data_format='channels_last', name='preact_conv1x1',
                   return_preact=False):
    return PreActConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=1, strides=strides, padding=0,
                        data_format=data_format, name=name, return_preact=return_preact)


def preact_conv3x3(in_channels, out_channels, strides=1, data_format='channels_last', name='preact_conv3x3',
                   return_preact=False):
    return PreActConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, strides=strides, padding=1,
                        data_format=data_format, name=name, return_preact=return_preact)


class DenseUnit(tf.keras.Model):
    def __init__(self, in_channels, bn_size=4, growth_rate=32, dropout_rate=0, data_format='channels_last',
                 name='dense_block'):
        super(DenseUnit, self).__init__(name=name)
        self.concat = Concatenate(axis=-1 if data_format == 'channels_last' else 1, name=name + '/concat')
        self.conv1 = preact_conv1x1(in_channels=in_channels, out_channels=bn_size * growth_rate,
                                    data_format=data_format, name=name + '/conv1')
        self.conv2 = preact_conv3x3(in_channels=bn_size * growth_rate, out_channels=growth_rate,
                                    data_format=data_format, name=name + '/conv2')
        self.dropout = Dropout(rate=dropout_rate, name=name + '/dropout') if dropout_rate > 0 else None

    def call(self, x, training=False):
        if isinstance(x, list):
            if len(x) == 1:
                x = x[0]
            else:
                x = self.concat(x)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        if self.dropout:
            x = self.dropout(x, training=training)
        return x


class DenseBlock(tf.keras.Model):
    def __init__(self, in_channels, num_layers, bn_size, growth_rate, dropout_rate=0, data_format='channels_last',
                 name='dense_block'):
        super(DenseBlock, self).__init__(name=name)
        self.units = [
            DenseUnit(in_channels=in_channels + i * growth_rate, bn_size=bn_size, growth_rate=growth_rate,
                       dropout_rate=dropout_rate, data_format=data_format, name=name + '/dense_unit{}'.format(i)
            ) for i in range(num_layers)
        ]
        self.concat = Concatenate(axis=-1 if data_format == 'channels_last' else 1, name=name + '/concat')


    def call(self, x, training=False):
        features = [x]

        for unit in self.units:
            new_features = unit(features, training=training)
            features.append(new_features)

        x = self.concat(features)
        return x


class TransitBlock(tf.keras.Model):
    def __init__(self, in_channels, out_channels, data_format='channels_last', name='transition'):
        super(TransitBlock, self).__init__(name=name)
        self.model = tf.keras.Sequential([
            BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/bn'),
            ReLU(name=name + '/activ'),
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, strides=1, padding=0,
                   use_bias=False, data_format=data_format, name=name + '/conv'),
            AveragePooling2D(pool_size=(2, 2), strides=2, data_format=data_format, name=name + '/pool')
        ])

    def call(self, x, training=False):
        return self.model(x, training=training)


class DenseNet(tf.keras.Model):
    """Implementation of DenseNet.

    reference: https://arxiv.org/abs/1608.06993.
    """
    def __init__(self, bn_size=4, growth_rate=32, blocks=(6, 12, 24, 16), in_channels=3, init_channels=64,
                 dropout_rate=.1, num_classes=1000, data_format='channels_last', name='densenet'):
        super(DenseNet, self).__init__(name=name)
        self.init = tf.keras.Sequential(
            layers = [
                Conv2d(in_channels=in_channels, out_channels=init_channels, kernel_size=7, strides=2, padding=3,
                       use_bias=False, data_format=data_format, name = name + '/init/conv0'),
                BatchNormalization(axis=-1 if data_format=='channels_last' else 1, name=name + '/init/bn'),
                ReLU(name=name + '/init/activ'),
                MaxPool2d(pool_size=3, strides=2, padding=1, data_format=data_format, name=name + '/init/pool')
            ], name=name + '/init'
        )

        self.dense_blocks = tf.keras.Sequential(name=name + '/dense_blocks')

        num_channels = init_channels
        for i, num_layers in enumerate(blocks, 1):
            dense_block = DenseBlock(
                in_channels=num_channels,
                num_layers=num_layers,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_rate=dropout_rate,
                data_format=data_format,
                name=name + 'dense_blocks/dense_block{}'.format(i)
            )
            self.dense_blocks.add(dense_block)
            num_channels += num_layers * growth_rate

            if i != len(blocks):
                transit_block = TransitBlock(
                    in_channels=num_channels,
                    out_channels=num_channels // 2,
                    data_format=data_format,
                    name= name + 'dense_blocks/transit_block{}'.format(i)
                )
                num_channels = num_channels // 2
                self.dense_blocks.add(transit_block)

        self.features = tf.keras.Sequential(
            layers=[
                BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/features/bn'),
                ReLU(name = name + '/features/activ'),
                GlobalAveragePooling2D(data_format=data_format, name=name + '/features/final_pool')
            ], name=name + '/features'
        )

        self.classifier = Dense(units=num_classes, name=name + '/classifier')


    def call(self, x, training=False):
        x = self.init(x, training=training)
        x = self.dense_blocks(x, training=training)
        x = self.features(x, training=training)
        x = self.classifier(x)
        return x


def get_densenet121():
    return DenseNet(growth_rate=32, init_channels=64, blocks=[6, 12, 24, 16])


def get_densenet161():
    return DenseNet(growth_rate=48, init_channels=96, blocks=[6, 12, 36, 24])


def get_densenet169():
    return DenseNet(growth_rate=32, init_channels=64, blocks=[6, 12, 32, 32])


def get_densenet201():
    return DenseNet(growth_rate=32, init_channels=64, blocks=[6, 12, 48, 32])


def _test_preact_conv():
    x = tf.random.uniform((32, 56, 56, 64))
    m = preact_conv1x1(64, 128)
    m2 = preact_conv3x3(128, 32)
    o1 = m(x)
    assert o1.shape == (32, 56, 56, 128)
    assert get_num_params(m) == 64 * 2 + 128 * 64 * (1 * 1 + 0)
    o2 = m2(o1)
    assert o2.shape == (32, 56, 56, 32)
    assert get_num_params(m2) == 128 * 2 + 128 * 32 * (3 * 3 + 0)


def _test_denseunit():
    x = tf.random.uniform((32, 56, 56, 64))
    bn_size, growth_rate = 4, 32
    m = DenseUnit(64, bn_size=bn_size, growth_rate=growth_rate, dropout_rate=.1)
    o = m(x)
    assert o.shape == ((32, 56, 56, growth_rate))
    conv1_params = 64 * 2 + 64 * (bn_size * growth_rate) * (1 * 1 + 0)
    conv2_params = (bn_size * growth_rate) * 2 + (bn_size * growth_rate) * growth_rate * (3 * 3 + 0)
    num_params = conv1_params + conv2_params
    assert get_num_params(m) == num_params


def _test_denseblock():
    x = tf.random.uniform((32, 56, 56, 64))
    bn_size, growth_rate = 4, 32
    m = DenseBlock(in_channels=64, num_layers=6, bn_size=bn_size, growth_rate=growth_rate)
    o = m(x)
    assert o.shape == (32, 56, 56, 64 + 6 * 32)
    assert get_num_params(m) == 335040


def _test_trasitblock():
    x = tf.random.uniform((32, 56, 56, 256))
    m = TransitBlock(in_channels=256, out_channels=128)
    o = m(x)
    assert o.shape == (32, 28, 28, 128)
    assert get_num_params(m) == 256 * 2 + 256 * 128 * (1 * 1 + 0)


def _test_densenet():
    x = tf.random.uniform((32, 224, 224, 3))
    models = [get_densenet121, get_densenet161, get_densenet169, get_densenet201]
    num_params = [7978856, 28681000, 14149480, 20013928]
    for model, num_param in zip(models, num_params):
        m = model()
        o = m(x)
        assert o.shape == (32, 1000)
        assert get_num_params(m) == num_param


if __name__ == '__main__':
    _test_preact_conv()
    _test_denseunit()
    _test_denseblock()
    _test_trasitblock()
    _test_densenet()


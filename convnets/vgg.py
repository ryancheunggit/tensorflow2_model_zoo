import tensorflow as tf
from common import Conv2d, MaxPool2d, Flatten, get_num_params
from tensorflow.keras.layers import BatchNormalization

__all__ = ['VGG', 'construct_vgg_net', 'vgg11', 'vgg13', 'vgg16', 'vgg19']


class ConvBlock(tf.keras.Model):
    """Conv2d -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding, data_format, name='conv_block'):
        super(ConvBlock, self).__init__()
        self.conv2d = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            data_format=data_format,
            name=name + '/conv'
        )
        self.batchnorm = BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/bn')
        self.activation = tf.keras.layers.ReLU(name=name + '/activ')

    def call(self, x, training=False):
        return self.activation(self.batchnorm(self.conv2d(x), training=training))


class DenseBlock(tf.keras.Model):
    """Fully Connected -> Dropout -> ReLU"""
    def __init__(self, out_features, dropout_rate, name='dense'):
        super(DenseBlock, self).__init__()
        self.fc = tf.keras.layers.Dense(units=out_features, name=name + '/fc')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name=name + '/dropout')
        self.relu = tf.keras.layers.ReLU(name=name + '/activ')

    def call(self, x, training=True):
        return self.relu(self.dropout(self.fc(x), training=training))


class Classifier(tf.keras.Model):
    """Dense -> Dense -> Dense"""
    def __init__(self, hidden_size=4096, dropout_rate=.5, num_classes=1000):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size,
        self.num_classes = num_classes
        self.classifier = tf.keras.Sequential([
            DenseBlock(out_features=hidden_size, dropout_rate=dropout_rate, name='fc1'),
            DenseBlock(out_features=hidden_size, dropout_rate=dropout_rate, name='fc2'),
            DenseBlock(out_features=num_classes, dropout_rate=dropout_rate, name='fc3')
        ])

    def call(self, x, training=True):
        return self.classifier(x, training=training)


class VGG(tf.keras.Model):
    """Implementation of VGG.

    reference: https://arxiv.org/abs/1409.1556

    added batch normalization to conv layers, dropouts to fully connected layers.
    """
    def __init__(self,
            channels,
            in_channels=3,
            hidden_size=4096,
            dropout_rate=.5,
            num_classes=1000,
            data_format='channels_last',
            **kwargs):
        super(VGG, self).__init__()
        features = tf.keras.Sequential(name='feature_extractor')
        for i, channels_per_stage in enumerate(channels):
            for j, out_channels in enumerate(channels_per_stage):
                features.add(
                    ConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        strides=1,
                        padding=1,
                        data_format=data_format,
                        name='features/stage{}/conv{}'.format(i, j)
                    )
                )
                in_channels = out_channels
            features.add(
                MaxPool2d(
                    pool_size=2,
                    strides=2,
                    padding=0,
                    ceil_mode=False,
                    data_format=data_format,
                    name='features/stage{}/pool'.format(i)
                )
            )
        features.add(Flatten(data_format=data_format))
        self.features = features
        self.classifier = Classifier(hidden_size=hidden_size, dropout_rate=dropout_rate, num_classes=num_classes)

    def call(self, x, training=True):
        features = self.features(x, training=training)
        logits = self.classifier(features, training=training)
        return logits


def construct_vgg_net(layers_per_stage=None, channels_per_stage=None, **kwargs):
    layers_per_stage = [2, 2, 3, 3, 3] if not layers_per_stage else layers_per_stage
    channels_per_stage = [64, 128, 256, 512, 512] if not channels_per_stage else channels_per_stage
    assert len(layers_per_stage) == len(channels_per_stage)
    channels = [[nc] * nl for (nc, nl) in zip(channels_per_stage, layers_per_stage)]
    return VGG(channels=channels, **kwargs)


def vgg11():
    return construct_vgg_net(layers_per_stage=[1, 1, 2, 2, 2])


def vgg13():
    return construct_vgg_net(layers_per_stage=[2, 2, 2, 2, 2])


def vgg16():
    return construct_vgg_net(layers_per_stage=[2, 2, 3, 3, 3])


def vgg19():
    return construct_vgg_net(layers_per_stage=[2, 2, 4, 4, 4])


def _test_vgg():
    x = tf.random.uniform((32, 224, 224, 3))
    model = vgg11()
    out = model(x, training=True)
    assert out.shape == (32, 1000)
    assert get_num_params(model) == 132866088
    model = vgg13()
    out = model(x, training=True)
    assert out.shape == (32, 1000)
    assert get_num_params(model) == 133050792
    model = vgg16()
    out = model(x, training=True)
    assert out.shape == (32, 1000)
    assert get_num_params(model) == 138361768
    model = vgg19()
    out = model(x, training=True)
    assert out.shape == (32, 1000)
    assert get_num_params(model) == 143672744


if __name__ == '__main__':
    _test_vgg()

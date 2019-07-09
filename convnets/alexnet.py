import numpy as np
import tensorflow as tf
from common import Conv2d, MaxPool2d, Flatten


class ConvBlock(tf.keras.Model):
    """Conv2d -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding, data_format, name='conv_block'):
        super(ConvBlock, self).__init__()
        self.conv2d = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=True,
            data_format=data_format,
            name=name + '/conv'
        )
        self.activation = tf.keras.layers.ReLU(name=name + '/relu')

    def call(self, x):
        return self.activation(self.conv2d(x))


class DenseBlock(tf.keras.Model):
    """Dropout -> Fully Connected -> ReLU"""
    def __init__(self, out_features, dropout_rate, name='dense'):
        super(DenseBlock, self).__init__()
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name=name + '/dropout')
        self.fc = tf.keras.layers.Dense(units=out_features, name=name + '/fc')
        self.relu = tf.keras.layers.ReLU(name=name + '/relu')

    def call(self, x, training=True):
        return self.relu(self.fc(self.dropout(x, training=training)))


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


class AlexNet(tf.keras.Model):
    """Implementation of AlexNet.

    reference: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    """
    def __init__(self,
            channels,
            kernel_sizes,
            strides,
            paddings,
            in_channels=3,
            hidden_size=4096,
            dropout_rate=.5,
            num_classes=1000,
            data_format='channels_last',
            **kwargs):
        super(AlexNet, self).__init__()
        model = tf.keras.Sequential()
        for i, channels_per_stage in enumerate(channels):
            for j, out_channels in enumerate(channels_per_stage):
                model.add(
                    ConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[i][j],
                        strides=strides[i][j],
                        padding=paddings[i][j],
                        data_format=data_format,
                        name='features/stage{}/conv{}'.format(i, j)
                    )
                )
                in_channels = out_channels
            model.add(
                MaxPool2d(
                    pool_size=3,
                    strides=2,
                    padding=0,
                    ceil_mode=False,
                    data_format=data_format,
                    name='features/stage{}/pool'.format(i)
                )
            )
        model.add(Flatten(data_format=data_format))
        model.add(Classifier(hidden_size=hidden_size, dropout_rate=dropout_rate, num_classes=num_classes))
        self.model = model

    def call(self, x, training=True):
        return self.model(x, training=training)


def construct_alex_net(**kwargs):
    channels = [[64], [192], [384, 256, 256]]
    kernel_sizes = [[11], [5], [3, 3, 3]]
    strides = [[4], [1], [1, 1, 1]]
    paddings = [[2], [2], [1, 1, 1]]
    model = AlexNet(channels, kernel_sizes, strides, paddings, **kwargs)
    return model


def _get_num_params(module):
    """Calculate the number of parameters of a neural network module."""
    return np.sum([np.prod(v.get_shape().as_list()) for v in module.variables])


def _test_alexnet():
    model = construct_alex_net()
    model.build(input_shape=(None, 224, 224, 3))
    # model.compile(optimizer=tf.keras.optimizers.Adam())
    x = tf.random.uniform((32, 224, 224, 3))
    out = model(x, training=True)
    assert out.shape == (32, 1000)
    assert _get_num_params(model) == 61100840


if __name__ == '__main__':
    _test_alexnet()

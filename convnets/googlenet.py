import tensorflow as tf
from common import Conv2d, MaxPool2d, Flatten, get_num_params
from tensorflow.keras.layers import BatchNormalization

__all__ = ['GoogleNet']


class ConvBlock(tf.keras.Model):
    """Conv2d -> BatchNormalization -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 data_format='channels_last', name='conv_block', **kwargs):
        super(ConvBlock, self).__init__()
        self.conv2d = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bias=False,
            data_format=data_format,
            name=name + '/conv',
            **kwargs)
        self.batchnorm = BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/bn')
        self.activation = tf.keras.layers.ReLU(name=name + '/activ')

    def call(self, x, training=False):
        return self.activation(self.batchnorm(self.conv2d(x), training=training))


class InceptionBlock(tf.keras.Model):
    """Inception module."""
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                 data_format='channels_last', name='inception'):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, ch1x1, 1, data_format, name=name + '/1x1conv')

        self.branch2 = tf.keras.Sequential([
            ConvBlock(in_channels, ch3x3red, 1, data_format, name=name + '/3x3reduction'),
            ConvBlock(ch3x3red, ch3x3, 3, data_format, name=name + '/3x3conv', padding=1)
        ])

        self.branch3 = tf.keras.Sequential([
            ConvBlock(in_channels, ch5x5red, 1, data_format, name=name + '/5x5reduction'),
            ConvBlock(ch5x5red, ch5x5, 5, data_format, name=name + '/5x5conv', padding=2)
        ])

        self.branch4 = tf.keras.Sequential([
            MaxPool2d(pool_size=3, strides=1, padding=1, ceil_mode=True, data_format=data_format,
                      name=name + '/3x3pool'),
            ConvBlock(in_channels, pool_proj, 1, data_format, name=name +'/pool_projection')
        ])

        self.concat = tf.keras.layers.Concatenate(
            axis=-1 if data_format == 'channels_last' else 1,
            name=name + 'concat'
        )


    def call(self, x, training=False):
        branch1 = self.branch1(x, training=training)
        branch2 = self.branch2(x, training=training)
        branch3 = self.branch3(x, training=training)
        branch4 = self.branch4(x, training=training)
        return self.concat([branch1, branch2, branch3, branch4])


class InceptionAux(tf.keras.Model):
    def __init__(self, in_channels, num_classes, data_format, name='inception_aux'):
        super(InceptionAux, self).__init__()
        self.avg_pool = tf.keras.layers.AveragePooling2D(3, data_format=data_format)
        self.conv = ConvBlock(in_channels, 128, 1, data_format, name=name + '/1x1conv')
        self.flatten = Flatten(data_format)
        self.fc1 = tf.keras.layers.Dense(1024, name=name + '/fc1')
        self.dropout = tf.keras.layers.Dropout(rate=.5, name=name + '/dropout')
        self.activation = tf.keras.layers.ReLU()
        self.fc2 = tf.keras.layers.Dense(num_classes, name=name + '/logits')

    def call(self, x, training=True):
        x = self.avg_pool(x)  # N x 4 x 4 x 512 for aux1  N x 4 x 4 x 528 for aux2
        x = self.conv(x)  # N x 4 x 4 x 128
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x


class GoogleNet(tf.keras.Model):
    """Implementation of GoogleNet(Inception V1).

    reference: http://arxiv.org/abs/1409.4842
    """
    def __init__(self, in_channels=3, data_format='channels_last', aux_logits=True, num_classes=1000):
        super(GoogleNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = ConvBlock(in_channels, 64, 7, data_format, name='conv1', strides=2, padding=3)
        self.maxpool1 = MaxPool2d(3, strides=2, ceil_mode=True, data_format=data_format, name='pool1')
        self.conv2 = ConvBlock(64, 64, 1, data_format, name='conv2')
        self.conv3 = ConvBlock(64, 192, 3, data_format, name='conv3')
        self.maxpool2 = MaxPool2d(3, strides=2, ceil_mode=True, data_format=data_format, name='pool2')

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32, data_format, name='inception3a')
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64, data_format, name='inception3b')
        self.maxpool3 = MaxPool2d(3, strides=2, ceil_mode=True, data_format=data_format, name='pool3')

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64, data_format, name='inception4a')
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64, data_format, name='inception4b')
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64, data_format, name='inception4c')
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64, data_format, name='inception4d')
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128, data_format, name='inception4e')
        self.maxpool4 = MaxPool2d(2, strides=2, ceil_mode=True, data_format=data_format, name='pool4')

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128, data_format, name='inception5a')
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128, data_format, name='inception5b')

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D(data_format, name='avgpool')

        self.aux_classifier1 = InceptionAux(512, num_classes, data_format, name='aux1')
        self.aux_classifier2 = InceptionAux(528, num_classes, data_format, name='aux2')

        self.classifier = tf.keras.Sequential([
                tf.keras.layers.Dropout(rate=.2, name='classifier/dropout'),
                tf.keras.layers.Dense(num_classes, name='classifier/fc')
        ])

    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.maxpool1(x)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.maxpool2(x)  # N x 28 x 28 x 192

        x = self.inception3a(x, training=training)
        x = self.inception3b(x, training=training)
        x = self.maxpool3(x)  # N x 13 x 13 x 480

        x = self.inception4a(x, training=training)
        if training and self.aux_logits:
            aux1 = self.aux_classifier1(x)
        x = self.inception4b(x, training=training)
        x = self.inception4c(x, training=training)
        x = self.inception4d(x, training=training)
        x = self.inception4e(x, training=training)
        x = self.maxpool4(x)  # N x 7 x 7 x 832

        x = self.inception5a(x, training=training)
        if training and self.aux_logits:
            aux2 = self.aux_classifier2(x)
        x = self.inception5b(x, training=training)

        x = self.avgpool(x)  # N x 1024
        x = self.classifier(x, training=training)

        if training and self.aux_logits:
            return (x, aux1, aux2)
        return x



def _test_inception():
    x = tf.random.uniform((32, 28, 28, 256))
    m = InceptionBlock(256, 128, 64, 192, 64, 96, 64)
    o = m(x)
    assert o.shape == (32, 28, 28, 480)


def _test_aux():
    x = tf.random.uniform((32, 14, 14, 512))
    m = InceptionAux(512, 1000, 'channels_last')
    o = m(x)
    assert o.shape == (32, 1000)


def _test_googlenet():
    x = tf.random.uniform((32, 224, 224, 3))
    m = GoogleNet()
    o, a1, a2 = m(x, training=True)
    assert o.shape == a1.shape == a2.shape == (32, 1000)
    o = m(x)
    assert o.shape == (32, 1000)
    assert get_num_params(m) == 11851864

if __name__ == '__main__':
   _test_inception()
   _test_aux()
   _test_googlenet()

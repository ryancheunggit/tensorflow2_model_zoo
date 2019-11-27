import tensorflow as tf
from common import EmbedFeatures, FullyConnectedNetwork, LinearModel


class CompressedInteractionNetwork(tf.keras.Model):
    """CIN part for xDeepFM.

    The structure of CIN is very similar to the RNN, where the ouputs of a hidden layer depends on the outputs from
    the previous layer(h) and an additonal input(x0).

    And the hidden layer is Conv1D that slides through the factor dimension to compress the interaction dimension.
    """
    def __init__(self, cin_hidden_sizes, split=True, name='cin'):
        super(CompressedInteractionNetwork, self).__init__(name=name)
        self.num_layers = len(cin_hidden_sizes)
        self.split = split
        self.conv_layers = [
            tf.keras.layers.Conv1D(filters=cin_hidden_size, kernel_size=1, activation='relu',
                                   data_format='channels_last', name=name + '/conv{}'.format(i))
            for i, cin_hidden_size in enumerate(cin_hidden_sizes)
        ]
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.fc = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        batch_size, factor_dim = tf.shape(x)[0], tf.shape(x)[2]
        states = []
        x0, h = tf.expand_dims(x, 2), x
        for i in range(self.num_layers):
            x = tf.multiply(x0, tf.expand_dims(h, 1))      # out product
            x = tf.reshape(x, shape=(batch_size, -1, factor_dim))
            # have to swap channel, as tf2beta don't support channels_first for Conv1D yet!
            x = tf.transpose(x, perm=[0, 2, 1])            # to bs x factor x interaction
            x = self.conv_layers[i](x)
            x = tf.transpose(x, perm=[0, 2, 1])            # to bs x compressed interaction x factor
            if self.split and i != self.num_layers - 1:
                split_size = tf.shape(x)[1] // 2
                x, h = tf.split(x, [split_size, split_size], axis=1)
            else:
                h = x
            states.append(x)
        states = self.concat(states)
        return self.fc(tf.reduce_sum(states, axis=2))


class ExtremeDeepFactorizationMachine(tf.keras.Model):
    """Implementation of xDeepFM

    reference: https://arxiv.org/abs/1803.05170
    """
    def __init__(self, feature_cards, factor_dim, fnn_hidden_sizes, cin_hidden_sizes, dropout_rate=.1, split=True,
                 name='xdeepfm'):
        super(ExtremeDeepFactorizationMachine, self).__init__(name=name)
        self.linear = LinearModel(feature_cards, name=name + '/linear_model')
        self.embedding = EmbedFeatures(feature_cards, factor_dim, name=name + '/feature_embedding')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_first')
        self.nn = FullyConnectedNetwork(units=fnn_hidden_sizes, dropout_rate=dropout_rate, name=name + '/fcn')
        self.cin = CompressedInteractionNetwork(cin_hidden_sizes, split, name=name + '/cin')

    def call(self, x, training=False):
        factors = self.embedding(x)
        features = self.flatten(factors)

        linear_out = self.linear(x, training=training)
        fnn_out = self.nn(features, training=training)
        cin_out = self.cin(factors, training=training)
        return linear_out + fnn_out + cin_out


def _test_cin():
    x = tf.random.uniform((32, 20, 5))
    cin = CompressedInteractionNetwork([100, 50, 10], split=True)
    o = cin(x)
    assert o.shape == (32, 1)


def _test_xdfm():
    x = tf.convert_to_tensor([[1, 2, 2, 1, 5], [0, 3, 2, 2, 4], [2, 2, 2, 1, 1]])
    m = ExtremeDeepFactorizationMachine([3, 4, 5, 4, 6], 4, [20, 10, 1], [10, 5], .1, False)
    o = m(x)
    assert o.shape == (32, 1)


if __name__ == '__main__':
    _test_cin()
    _test_xdfm()

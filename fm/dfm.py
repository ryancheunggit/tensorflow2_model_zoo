import tensorflow as tf
from common import EmbedFeatures, FullyConnectedNetwork, LinearModel


class DeepFM(tf.keras.Model):
    """Implementation of DeepFM

    Reference: https://arxiv.org/abs/1703.04247

    It is basically FM + FNN
    """
    def __init__(self, feature_cards, factor_dim, hidden_sizes, dropout_rate=.1, name='deepfm'):
        super(DeepFM, self).__init__(name=name)
        self.linear = LinearModel(feature_cards, name=name + '/linear_model')
        self.embedding = EmbedFeatures(feature_cards, factor_dim, name=name + '/feature_embedding')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_first')
        self.nn = FullyConnectedNetwork(units=hidden_sizes, dropout_rate=dropout_rate, name=name + '/fcn')

    def call(self, x, training=False):
        factors = self.embedding(x)
        # FM part
        linear_out = self.linear(x)
        sum_of_squares = tf.reduce_sum(tf.pow(factors, 2), 1)
        square_of_sums = tf.pow(tf.reduce_sum(factors, 1), 2)
        interaction_out = .5 * tf.reduce_sum(square_of_sums - sum_of_squares, 1, keepdims=True)
        fm_out = linear_out + interaction_out

        # FNN part
        features = self.flatten(factors)
        fnn_out = self.nn(features, training=training)
        return fm_out + fnn_out


def _test_dfm():
    x = [[1, 2, 2], [0, 3, 2]]
    m = DeepFM([3, 4, 5], 2, [5, 3, 1], .1)
    o = m(x)
    assert o.shape == (2, 1)


if __name__ == '__main__':
    _test_dfm()



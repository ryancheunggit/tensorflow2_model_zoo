import tensorflow as tf
from common import EmbedFeatures, FullyConnectedNetwork


class FMNeuralNetwork(tf.keras.Model):
    """Implementation of Factorization-machine supported Neural Networks.

    Reference: https://arxiv.org/abs/1601.02376

    It flattens feature embeddings and pass through fully connected layers.
    """
    def __init__(self, feature_cards, factor_dim, hidden_sizes, dropout_rate=.1, name='neural_factorization_machine'):
        super(FMNeuralNetwork, self).__init__(name=name)
        self.embedding = EmbedFeatures(feature_cards, factor_dim, name=name + '/feature_embedding')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_first')
        self.nn = FullyConnectedNetwork(units=hidden_sizes, dropout_rate=dropout_rate, name=name + '/fcn')

    def call(self, x, training=False):
        features = self.flatten(self.embedding(x))
        return self.nn(features, training=training)


def _test_fnn():
    x = [[1, 2, 2], [0, 3, 2]]
    m = FMNeuralNetwork([3, 4, 5], 2, [5, 3, 1], .1)
    o = m(x)
    assert o.shape == (2, 1)


if __name__ == '__main__':
    _test_fnn()

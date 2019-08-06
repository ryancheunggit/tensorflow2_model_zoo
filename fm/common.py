import numpy as np
import tensorflow as tf


class LinearModel(tf.keras.Model):
    """The linear part of the FM model.

    w_0 + \Sigma_{i=1}^n {w_ix_i}

    The input to this module is the same as the EmbedFeatures module, see that part for detail.
    """
    def __init__(self, feature_cards, name='linear_model'):
        super(LinearModel, self).__init__(name=name)
        self.bias = tf.random.uniform((1,))
        self.linear = tf.keras.layers.Embedding(input_dim=sum(feature_cards), output_dim=1)
        self.offsets = tf.constant(np.concatenate(([0], np.cumsum(feature_cards)[:-1])))

    def call(self, x, training=False):
        x = x + self.offsets
        return self.bias + tf.reduce_sum(self.linear(x), axis=1)


class EmbedFeatures(tf.keras.Model):
    """Embed encoded features.

    Each input to the embedding module is encoded values of its features.
    Each feature is indexed from 0.
    For continuous features, we would need to bin and encode first.

    For example, if our dataset has three features, value and (encoding):
        feature 1: gender - F(0), M(1), O(2)
        feature 2: edu - highschool(0), bachelor(1), master(2), phd(3)
        feature 3: age - 18-(0), 18-24(1), 25-40(2), 40-65(3), 65+(4)
    and we want to embed features to dimension k=2.

    To initialize the embedding, we would pass feature_cards = [3, 4, 5] and factor_dim = 2
    To retrive embeddings for our examples:
        an input example that represents a (male, master, 26) would be [1, 2, 2]
        an input example that represents a (female, phd, 35) would be [0, 3, 2]
    """
    def __init__(self, feature_cards, factor_dim, name='embedding'):
        super(EmbedFeatures, self).__init__(name=name)
        self.embedding = tf.keras.layers.Embedding(input_dim=sum(feature_cards), output_dim=factor_dim)
        self.offsets = tf.constant(np.concatenate(([0], np.cumsum(feature_cards)[:-1])))

    def call(self, x, training=False):
        x = x + self.offsets
        embedded = self.embedding(x)
        return embedded


def _test_embedding_features():
    x = [[1, 2, 2], [0, 3, 2]]
    m = EmbedFeatures([3, 4, 5], 2)
    o = m(x)
    assert o.shape == (2, 3, 2)


def _test_linear_model():
    x = [[1, 2, 2], [0, 3, 2]]
    m = LinearModel([3, 4, 5])
    o = m(x)
    assert o.shape == (2, 1)


if __name__ == '__main__':
    _test_embedding_features()
    _test_linear_model()

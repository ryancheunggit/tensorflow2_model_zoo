import tensorflow as tf
from common import LinearModel, FieldAwareEmbedFeatures, FullyConnectedNetwork


class FieldAwareNeuralFactorizationMachine(tf.keras.Model):
    """Implementation of Field-aware Neural Factorization Machines.

    Reference: https://arxiv.org/abs/1902.09096

    extend nfm to operate on field aware interactions
    """
    def __init__(self, feature_cards, factor_dim, hidden_sizes, dropout_rate=.1, name='fnfm'):
        super(FieldAwareNeuralFactorizationMachine, self).__init__(name=name)
        self.num_features = len(feature_cards)
        self.embeddings = FieldAwareEmbedFeatures(feature_cards, factor_dim,
                                                  name=name + '/field_aware_feature_embedding')
        self.linear = LinearModel(feature_cards, name=name + '/linear_model')
        self.nn = FullyConnectedNetwork(units=hidden_sizes, dropout_rate=dropout_rate, name=name + '/fcn')

    def call(self, x, training=False):
        batch_size, num_features = int(tf.shape(x)[0]), self.num_features
        num_interactions = num_features * (num_features - 1) // 2

        linear_out = self.linear(x)

        factors_i = self.embeddings(x)
        factors_j = tf.transpose(factors_i, [0, 2, 1, 3])
        interactions = tf.reduce_sum(tf.multiply(factors_i, factors_j), axis=-1)
        mask = tf.ones_like(interactions)
        mask = tf.cast(tf.linalg.band_part(mask, 0, -1) - tf.linalg.band_part(mask, 0, 0), dtype=tf.bool)
        interactions = tf.boolean_mask(interactions, mask)
        interactions = tf.reshape(interactions, shape=[batch_size, num_interactions])
        interaction_out = self.nn(interactions, training=training)

        return linear_out + interaction_out


def _test_fnfm():
    x = [[1, 2, 2], [0, 3, 2]]
    m = FieldAwareNeuralFactorizationMachine([3, 4, 5], 2, [5, 3, 1], .1)
    o = m(x)
    assert o.shape == (2, 1)


if __name__ == '__main__':
    _test_fnfm()

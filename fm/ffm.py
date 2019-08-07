import tensorflow as tf
from common import LinearModel, FieldAwareEmbedFeatures


class FieldAwareFactorizationMachine(tf.keras.Model):
    """Implementation of Field-aware Factorization Machines.

    Reference: https://dl.acm.org/citation.cfm?id=2959134

    Difference with FM is that, each feature has #feautres latent factors. So when calculating interaction between
    feature_i and feature_j it is doing dot product between the jth factor of feature i and ith factor of feature j.
    """
    def __init__(self, feature_cards, factor_dim, name='ffm'):
        super(FieldAwareFactorizationMachine, self).__init__(name=name)
        self.factor_dim = factor_dim
        self.embeddings = FieldAwareEmbedFeatures(feature_cards, factor_dim,
                                                  name=name + '/field_aware_feature_embeddings')
        self.linear = LinearModel(feature_cards, name=name + '/linear_model')

    def call(self, x, training=False):
        batch_size, factor_dim = int(tf.shape(x)[0]), self.factor_dim
        linear_out = self.linear(x)
        factors_i = self.embeddings(x)
        factors_j = tf.transpose(factors_i, [0, 2, 1, 3])
        interactions = tf.reduce_sum(tf.multiply(factors_i, factors_j), -1)
        interaction_out = tf.expand_dims(tf.reduce_sum(
            tf.linalg.band_part(interactions, 0, -1) - tf.linalg.band_part(interactions, 0, 0),
            axis=(1, 2)
        ), axis=-1)
        return linear_out + interaction_out


def _test_ffm():
    x = tf.convert_to_tensor([[1, 2, 2], [0, 3, 2]], dtype='int32')
    m = FieldAwareFactorizationMachine([3, 4, 5], 2)
    o = m(x)
    assert o.shape == (2, 1)


if __name__ == '__main__':
    _test_ffm()

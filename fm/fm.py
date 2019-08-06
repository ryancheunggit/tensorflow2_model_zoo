import tensorflow as tf
from common import LinearModel, EmbedFeatures


class FactorizationMachine(tf.keras.Model):
    """Implementation of Factorization Machines.

    Degree is 2, unchangable. The input is discretized/Integer Encoded instead of real valued.
    See common.EmbedFeatures for detail on input configs.

    Reference: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    """
    def __init__(self, feature_cards, factor_dim, name='factorization_machine'):
        super(FactorizationMachine, self).__init__(name=name)
        self.embedding = EmbedFeatures(feature_cards, factor_dim, name=name + '/feature_embedding')
        self.linear = LinearModel(feature_cards, name=name + '/linear_model')

    def call(self, x, training=False):
        linear_out = self.linear(x)
        factors = self.embedding(x)
        sum_of_squares = tf.reduce_sum(tf.pow(factors, 2), 1)
        square_of_sums = tf.pow(tf.reduce_sum(factors, 1), 2)
        interaction_out = 0.5 * tf.reduce_sum(square_of_sums - sum_of_squares, 1, keepdims=True)
        return linear_out + interaction_out


def _test_fm():
    x = [[1, 2, 2], [0, 3, 2]]
    m = FactorizationMachine([3, 4, 5], 2)
    o = m(x)
    assert o.shape == (2, 1)


if __name__ == '__main__':
    _test_fm()

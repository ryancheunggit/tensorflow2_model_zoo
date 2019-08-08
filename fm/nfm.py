import tensorflow as tf
from common import LinearModel, EmbedFeatures, FullyConnectedNetwork


class NeuralFactorizationMachine(tf.keras.Model):
    """Implementation of Neural Factorization Machines.

    Reference: https://arxiv.org/abs/1708.05027

    Contrast to FNN, the fully connected network takes in the pooled pairwise interactions, and the model keeps the
    linear part.
    """
    def __init__(self, feature_cards, factor_dim, hidden_sizes, dropout_rate=.1, name='neural_factorization_machine'):
        super(NeuralFactorizationMachine, self).__init__(name=name)
        self.embedding = EmbedFeatures(feature_cards, factor_dim, name=name + '/feature_embedding')
        self.linear = LinearModel(feature_cards, name=name + '/linear_model')
        self.nn = FullyConnectedNetwork(units=hidden_sizes, dropout_rate=dropout_rate, name=name + '/fcn')

    def call(self, x, training=False):
        linear_out = self.linear(x)
        factors = self.embedding(x)
        sum_of_squares = tf.reduce_sum(tf.pow(factors, 2), 1)
        square_of_sums = tf.pow(tf.reduce_sum(factors, 1), 2)
        pooled_interactions = square_of_sums - sum_of_squares
        interaction_out = self.nn(pooled_interactions, training=training)
        return linear_out + interaction_out


def _test_nfm():
    x = [[1, 2, 2], [0, 3, 2]]
    m = NeuralFactorizationMachine([3, 4, 5], 2, [5, 3, 1], .1)
    o = m(x)
    assert o.shape == (2, 1)


if __name__ == '__main__':
    _test_nfm()

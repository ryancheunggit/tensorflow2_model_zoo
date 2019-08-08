import tensorflow as tf
from common import EmbedFeatures, FullyConnectedNetwork, LinearModel, MultiHeadAttention


class AutomaticFeatureInteraction(tf.keras.Model):
    """Implementation of AutoInt

    Reference: https://arxiv.org/abs/1810.11921

    Difference with FNN is that AutoInt using MultiHeadAttention to model feature interactions.
    """
    def __init__(self, feature_cards, factor_dim, n_heads, n_attentions, hidden_sizes, dropout_rate=.1, name='deepfm'):
        super(AutomaticFeatureInteraction, self).__init__(name=name)
        self.linear = LinearModel(feature_cards, name=name + '/linear_model')
        self.embedding = EmbedFeatures(feature_cards, factor_dim, name=name + '/feature_embedding')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_first')
        self.nn = FullyConnectedNetwork(units=hidden_sizes, dropout_rate=dropout_rate, name=name + '/fcn')
        self.attns = [
            MultiHeadAttention(n_heads, factor_dim, dropout_rate=dropout_rate, name=name + '/mhattn{}'.format(i))
            for i in range(n_attentions)
        ]
        self.attn_out = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        factors = self.embedding(x)                     # bs x num_features x factor_dim

        # linear part
        linear_out = self.linear(x)                     # bs x 1

        # FNN part
        features = self.flatten(factors)                # bs x num_features * factor_dim
        fnn_out = self.nn(features)                     # bs x 1

        # ATTN part
        cross_term = tf.transpose(factors, [0, 2, 1])   # bs x factor_dim x num_features
        for attn in self.attns:
            cross_term = attn(cross_term, cross_term, cross_term, training=training)
        cross_term = tf.transpose(cross_term, [0, 2, 1])
        cross_term = self.flatten(tf.nn.relu(cross_term))
        attn_out = self.attn_out(cross_term)
        return linear_out + fnn_out + attn_out


def _test_afi():
    x = tf.convert_to_tensor([[1, 2, 2, 1, 5], [0, 3, 2, 2, 4], [2, 2, 2, 1, 1]])
    m = AutomaticFeatureInteraction([3, 4, 5, 4, 6], 4, 2, 2, [5, 3, 1], .1)
    o = m(x)
    print(o.shape)
    assert o.shape == (3, 1)


if __name__ == '__main__':
    _test_afi()

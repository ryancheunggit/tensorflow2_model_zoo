import numpy as np
import tensorflow as tf


class LinearModel(tf.keras.Model):
    """The linear model.

    The input to this module is the same as the EmbedFeatures module, see that part for detail.
    """
    def __init__(self, feature_cards, name='linear_model'):
        super(LinearModel, self).__init__(name=name)
        self.bias = tf.random.uniform((1,))
        self.linear = tf.keras.layers.Embedding(input_dim=sum(feature_cards), output_dim=1)
        self.offsets = tf.constant(np.concatenate(([0], np.cumsum(feature_cards)[:-1])), dtype='int32')

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

    Example of encoded inputs:
        (male, master, 26) encoded as [1, 2, 2]
        (female, phd, 35) encoded as [0, 3, 2]
    """
    def __init__(self, feature_cards, factor_dim, name='embedding'):
        super(EmbedFeatures, self).__init__(name=name)
        self.embedding = tf.keras.layers.Embedding(input_dim=sum(feature_cards), output_dim=factor_dim)
        self.offsets = tf.constant(np.concatenate(([0], np.cumsum(feature_cards)[:-1])), dtype='int32')

    def call(self, x, training=False):
        x = x + self.offsets
        embedded = self.embedding(x)
        return embedded


class FieldAwareEmbedFeatures(tf.keras.Model):
    """Embed encoded features to num_features different latent factors.

    To get the ith feature's jth factor of example k in the batch, we would do: embedded[k, i, j, :]
    """
    def __init__(self, feature_cards, factor_dim, name='embedding'):
        super(FieldAwareEmbedFeatures, self).__init__(name=name)
        self.num_features = len(feature_cards)
        self.embeddings = [
            tf.keras.layers.Embedding(sum(feature_cards), factor_dim, name=name + '/embed_{}'.format(i))
            for i in range(self.num_features)
        ]
        self.offsets = tf.constant(np.concatenate(([0], np.cumsum(feature_cards)[:-1])), dtype='int32')

    def call(self, x, training=False):
        x = x + self.offsets
        embeddings = [tf.expand_dims(embedding(x), 2) for embedding in self.embeddings]
        return tf.concat(embeddings, 2)


class FullyConnectedNetwork(tf.keras.Model):
    """Fully connected feedforward network with optional dropouts."""
    def __init__(self, units, dropout_rate=.1, name='fcn'):
        super(FullyConnectedNetwork, self).__init__(name=name)
        layers = []
        for i, unit in enumerate(units):
            layers.append(tf.keras.layers.Dense(units=unit, name=name + '/fc{}'.format(i)))
            if dropout_rate > 0:
                layers.append(tf.keras.layers.Dropout(rate=dropout_rate, name=name + '/dropout{}'.format(i)))
        self.model = tf.keras.Sequential(layers)

    def call(self, x, training=False):
        return self.model(x, training=training)


def attention(value, key, query, mask=None, dropout=None, training=False):
    depth = query.shape[-1]
    logits = tf.matmul(query, key, transpose_b=True) / depth ** .5
    if mask:
        logits += mask * -1e9
    if dropout:
        logits = dropout(logits, training=training)
    alignment = tf.nn.softmax(logits, axis=-1)
    attended = tf.matmul(alignment, value)
    return attended


class MultiHeadAttention(tf.keras.Model):
    """Multi-Head Attention"""
    def __init__(self, n_heads, d_model, dropout_rate=.1, name='multihead_attention'):
        super(MultiHeadAttention, self).__init__(name=name)
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.depth = d_model // n_heads
        self.query_linear = tf.keras.layers.Dense(d_model)
        self.key_linear = tf.keras.layers.Dense(d_model)
        self.value_linear = tf.keras.layers.Dense(d_model)
        self.out_linear = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, value, key, query, mask=None, training=False):
        bs = tf.shape(query)[0]
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        query = tf.transpose(tf.reshape(query, (bs, -1, self.n_heads, self.depth)), [0, 2, 1, 3])
        key = tf.transpose(tf.reshape(key, (bs, -1, self.n_heads, self.depth)), [0, 2, 1, 3])
        value = tf.transpose(tf.reshape(value, (bs, -1, self.n_heads, self.depth)), [0, 2, 1, 3])

        attn = attention(query, key, value, mask=mask, dropout=self.dropout, training=training)
        attn = tf.transpose(attn, [0, 2, 1, 3])
        attn = tf.reshape(attn, (bs, -1, self.d_model))

        out = self.out_linear(attn)
        return out


def _test_linear_model():
    x = tf.convert_to_tensor([[1, 2, 2], [0, 3, 2]], dtype='int32')
    m = LinearModel([3, 4, 5])
    o = m(x)
    assert o.shape == (2, 1)


def _test_feature_embedding():
    x = tf.convert_to_tensor([[1, 2, 2], [0, 3, 2]], dtype='int32')
    m = EmbedFeatures([3, 4, 5], 2)
    o = m(x)
    assert o.shape == (2, 3, 2)


def _test_field_aware_feature_embedding():
    x = tf.convert_to_tensor([[1, 2, 2], [0, 3, 2]], dtype='int32')
    m = FieldAwareEmbedFeatures([3, 4, 5], 2)
    o = m(x)
    assert o.shape == (2, 3, 3, 2)


def _test_fully_connected():
    x = tf.random.uniform((32, 400))
    m = FullyConnectedNetwork([256, 128, 64, 1], .1)
    o = m(x)
    assert o.shape == (32, 1)


if __name__ == '__main__':
    _test_linear_model()
    _test_feature_embedding()
    _test_field_aware_feature_embedding()
    _test_fully_connected()

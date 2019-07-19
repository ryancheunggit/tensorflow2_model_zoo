import math
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dropout, Dense, ReLU, LayerNormalization


class Embedding(tf.keras.Model):
    '''Convert sequence of token ids to stacked dense vectors.'''
    def __init__(self, vocab_size, d_model, name='embed'):
        super(Embedding, self).__init__(name=name)
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def call(self, x):
        embedded = self.embedding(x) * math.sqrt(self.d_model)  # scale up
        return embedded


class PositionalEncoding(tf.keras.Model):
    '''Positional encoding

    positional encoding is used to add information about the relative position of tokens in sequence.
    The positional encoding vector has the same dimension(d_model) as the embeddings, so the two can be summed.

    This is fixed positional embedding, not learnable.
    pe_{pos, 2i} = sin(pos / exp(10000, 2i/d_model)
    pe_{pos, 2i + 1} = cos(pos / exp(10000, 2i/d_model)
    '''
    def __init__(self, d_model, max_seq_len=256, dropout_rate=.1, name='pe'):
        super(PositionalEncoding, self).__init__(name=name)
        self.d_model = d_model
        pe = np.zeros((max_seq_len, d_model))
        position = np.arange(0, max_seq_len, dtype='float')
        div_term = np.exp(np.arange(0, d_model, 2, dtype='float') * (- np.log(10000) / d_model))
        pe[:, 0::2] = np.sin(np.outer(position, div_term))
        pe[:, 1::2] = np.cos(np.outer(position, div_term))
        pe = np.expand_dims(pe, 0)
        self.pe = tf.constant(value=pe, dtype='float32', name='')
        self.dropout = Dropout(rate=dropout_rate)

    def call(self, x, training=False):
        seq_length = tf.shape(x)[1]
        x = x + tf.stop_gradient(self.pe[:, :seq_length])
        return self.dropout(x, training=training)


def attention(query, key, value, mask=None, dropout=None, training=False):
    """Scaled Dot-Product Attention.

    Attention(Q, K, V) = softmax_k((QK^T) / sqrt(depth))V
    """
    depth = query.shape[-1]

    # scaled matrix multiplication
    logits = tf.matmul(query, key, transpose_b=True) / math.sqrt(depth)

    if mask:
        logits += mask * -1e9

    if dropout:
        logits = dropout(logits, training=training)

    alignment = tf.nn.softmax(logits, axis=-1)
    attended = tf.matmul(alignment, value)
    return attended


class MultiHeadAttention(tf.keras.Model):
    '''Multi-Head Attention.

    Argument
    --------
        n_heads: number of attention head
        d_model: embedding dimension
        dropout_rate: dropout probablity
    '''
    def __init__(self, n_heads, d_model, dropout_rate=.1, name='mhattn'):
        super(MultiHeadAttention, self).__init__(name=name)
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.depth = d_model // n_heads
        self.query_linear = Dense(d_model)
        self.key_linear = Dense(d_model)
        self.value_linear = Dense(d_model)
        self.out_linear = Dense(d_model)
        self.dropout = Dropout(rate=dropout_rate)

    def call(self, query, key, value, mask=None, training=False):
        '''Forward pass of MultiHeadAttention.
        Argument
        --------
            query: of size (batch_size, query_length, model_size)
            key: of size (batch_size, key_length, model_size)
            value: of size (batch_size, value_length, model_size)
        '''
        bs = tf.shape(query)[0]

        # linear
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # split heads, convert to shape (batch_size, seq_length, n_heads, depth)
        query = tf.transpose(tf.reshape(query, (bs, -1, self.n_heads, self.depth)), [0, 2, 1, 3])
        key = tf.transpose(tf.reshape(key, (bs, -1, self.n_heads, self.depth)), [0, 2, 1, 3])
        value = tf.transpose(tf.reshape(value, (bs, -1, self.n_heads, self.depth)), [0, 2, 1, 3])

        # attention
        attn = attention(query, key, value, mask=mask, dropout=self.dropout, training=training)
        attn = tf.transpose(attn, [0, 2, 1, 3])

        # concatenate heads
        attn = tf.reshape(attn, (bs, -1, self.d_model))

        # final linear out
        out = self.out_linear(attn)
        return out


class PointwiseFeedForward(tf.keras.Model):
    def __init__(self, d_model, d_ff=768, dropout_rate=.1, name='pwffn'):
        super(PointwiseFeedForward, self).__init__(name=name)
        self.linear_1 = Dense(d_ff)
        self.activ = ReLU()
        self.dropout = Dropout(rate=dropout_rate)
        self.linear_2 = Dense(d_model)

    def call(self, x, training=False):
        x = self.linear_1(x)
        x = self.dropout(x, training=training)
        x = self.activ(x)
        x = self.linear_2(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    """Encoder Layer.

    There are two sublayers in each encoder layer:
        1. MultiHeadAttention
        2. PointwiseFeedForward
    The output of each sublayer is a residual connection of LayerNormalization(input + Sublayer(input)).
    """
    def __init__(self, n_heads, d_model, d_ff, dropout_rate=.1, epsilon=1e-6, name='encoder_layer'):
        super(EncoderLayer, self).__init__(name=name)
        self.multihead_attn = MultiHeadAttention(
            n_heads=n_heads, d_model=d_model, dropout_rate=dropout_rate, name=name + '/mha')
        self.feedforward = PointwiseFeedForward(
            d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate, name=name + '/ffn')
        self.layernorm1 = LayerNormalization(epsilon=epsilon, name=name + '/layernorm1')
        self.layernorm2 = LayerNormalization(epsilon=epsilon, name=name + '/layernorm2')
        self.dropout1 = Dropout(rate=dropout_rate, name=name + '/dropout1')
        self.dropout2 = Dropout(rate=dropout_rate, name=name + '/dropout2')

    def call(self, x, mask=None, training=False):
        attn_out = self.multihead_attn(x, x, x, mask, training=training)
        attn_out = self.dropout1(x, training=training)
        out1 = self.layernorm1(x + attn_out)

        ffn_out = self.feedforward(out1, training=training)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.layernorm2(out1 + ffn_out)
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """Decoder Layer.

    There are three sublayers in each decoder layer:
        1. MultiHeadAttention with lookahead_mask, it takes the same input as the coupled EncoderLayer
        2. MultiHeadAttention with padding_mask, it takes the output from the coupled EncoderLayer as input
        3. PointwiseFeedForward

    The output of each sublayer is a residual connection of LayerNormalization(input + Sublayer(input)).
    """
    def __init__(self, n_heads, d_model, d_ff, dropout_rate=.1, epsilon=1e-6, name='decoder'):
        super(DecoderLayer, self).__init__(name=name)
        self.multihead_attn1 = MultiHeadAttention(
            n_heads=n_heads, d_model=d_model, dropout_rate=dropout_rate, name=name + '/mha1')
        self.multihead_attn2 = MultiHeadAttention(
            n_heads=n_heads, d_model=d_model, dropout_rate=dropout_rate, name=name + '/mha2')
        self.feedforward = PointwiseFeedForward(
            d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate, name=name + '/ffn')
        self.layernorm1 = LayerNormalization(epsilon=epsilon, name=name + '/layernorm1')
        self.layernorm2 = LayerNormalization(epsilon=epsilon, name=name + '/layernorm2')
        self.layernorm3 = LayerNormalization(epsilon=epsilon, name=name + '/layernorm3')
        self.dropout1 = Dropout(rate=dropout_rate, name=name + '/dropout1')
        self.dropout2 = Dropout(rate=dropout_rate, name=name + '/dropout2')
        self.dropout3 = Dropout(rate=dropout_rate, name=name + '/dropout3')

    def call(self, x, enc_output, lookahead_mask=None, padding_mask=None, training=False):
        attn1 = self.multihead_attn1(x, x, x, mask=lookahead_mask, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.multihead_attn2(enc_output, enc_output, enc_output, mask=padding_mask, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_out = self.feedforward(out2, training=training)
        ffn_out = self.dropout3(ffn_out, training=training)
        out3 = self.layernorm3(ffn_out + out2)
        return out3


class Encoder(tf.keras.Model):
    def __init__(self, n_heads, d_model, num_layers, vocab_size, max_seq_len, d_ff=768, dropout_rate=.1,
                 name='encoder'):
        super(Encoder, self).__init__(name=name)
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size=vocab_size, d_model=d_model, name=name + '/embed')
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len,
                                                      dropout_rate=dropout_rate, name=name + '/pe')
        self.encoder_layers = [
            EncoderLayer(n_heads=n_heads, d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate,
                         name=name + '/encode_{}'.format(i))
            for i in range(num_layers)
        ]

    def call(self, x, mask=None, training=False):
        x = self.embedding(x)
        x = self.positional_encoding(x, training=training)
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, mask=mask, training=training)
        return x


class Decoder(tf.keras.Model):
    def __init__(self, n_heads, d_model, num_layers, vocab_size, max_seq_len, d_ff=768, dropout_rate=.1,
                 name='decoder'):
        super(Decoder, self).__init__(name=name)
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size=vocab_size, d_model=d_model, name=name + '/embed')
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len,
                                                      dropout_rate=dropout_rate, name=name + '/pe')
        self.decoder_layers = [
            DecoderLayer(n_heads=n_heads, d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate,
                         name=name + '/decode_{}'.format(i))
            for i in range(num_layers)
        ]

    def call(self, x, enc_output, lookahead_mask=None, padding_mask=None, training=False):
        x = self.embedding(x)
        x = self.positional_encoding(x, training=training)
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, enc_output, lookahead_mask=lookahead_mask, padding_mask=padding_mask,
                                       training=training)
        return x


class Transformer(tf.keras.Model):
    """Implementation of Transformer.

    reference: https://arxiv.org/abs/1706.03762
    """
    def __init__(self, n_heads, d_model, num_layers, d_ff, input_vocab_size, output_vocav_size, max_seq_len,
                 dropout_rate=.1, name='transformer'):
        super(Transformer, self).__init__(name=name)
        self.encoder = Encoder(n_heads, d_model, num_layers, input_vocab_size, max_seq_len, d_ff, dropout_rate,
                               name + '/encoder')
        self.decoder = Decoder(n_heads, d_model, num_layers, output_vocav_size, max_seq_len, d_ff, dropout_rate,
                               name + '/decoder')
        self.last_linear = Dense(output_vocav_size, name=name + '/last_linear')

    def call(self, inputs, targets, enc_padding_mask, lookahead_mask, dec_padding_mask, training=False):
        encoder_output = self.encoder(inputs, mask=enc_padding_mask, training=training)
        decoder_output = self.decoder(targets, encoder_output, lookahead_mask, dec_padding_mask, training=training)
        out = self.last_linear(decoder_output)
        return out


def _test_embedding():
    embedding = Embedding(vocab_size=50, d_model=3)
    x = np.random.randint(low=0, high=50, size=50).reshape((5, 10))
    o = embedding(x)
    assert o.shape == (5, 10, 3)


def _test_pe(plot=False):
    embedding = Embedding(50, 4)
    pe = PositionalEncoding(4, 15, .1)
    x1 = np.random.randint(low=0, high=50, size=50).reshape((5, 10))
    x2 = np.random.randint(low=0, high=50, size=45).reshape((3, 15))
    o1 = pe(embedding(x1))
    o2 = pe(embedding(x2))
    assert o1.shape == (5, 10, 4)
    assert o2.shape == (3, 15, 4)
    if plot:
        _plot_pe()


def _plot_pe():
    pe = PositionalEncoding(20)
    y = pe(tf.zeros((1, 60, 20)))
    f = plt.figure()
    plt.plot(np.arange(60), y.numpy()[0, :, 4:8])
    f.savefig('../plots/pe.png')
    plt.close()

    pos_encoding = PositionalEncoding(512, 50).pe
    # print(pos_encoding.shape)
    f = plt.figure()
    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    f.savefig('../plots/pe2.png')
    plt.close()


def _test_attention():
    # case 1: query matches the second key, attention would return the second value.
    q = tf.constant([[0, 10, 0]], dtype='float32')                                      # 1 x 3
    k = tf.constant([[10, 0, 0], [0, 10, 0], [0, 0, 10], [0, 0, 10]], dtype='float32')  # 4 x 3
    v = tf.constant([[1, 0], [10, 0], [100, 0], [1000, 0]], dtype='float32')            # 4 x 2
    o = attention(q, k, v)
    assert o.shape == (1, 2)
    t = np.array([[10, 0]])
    assert np.alltrue(o.numpy() == t)

    # case 2: query matches third and fourth key, attention would return average of the two values.
    q2 = tf.constant([[0, 0, 10]], dtype='float32')
    o2 = attention(q2, k, v)
    assert o2.shape == (1, 2)
    t2 = np.array([[(100 + 1000) / 2, 0]])
    assert np.alltrue(o2.numpy() == t2)

    # case 3
    q3 = tf.concat([q, q2], axis=0)
    o3 = attention(q3, k, v)
    assert o3.shape == (2, 2)
    t3 = np.vstack([t, t2])
    assert np.alltrue(o3.numpy() == t3)


def _test_multihead_attention():
    attn = MultiHeadAttention(n_heads=6, d_model=300)
    x = tf.random.uniform((32, 80, 300))
    out = attn(x, x, x)
    assert out.shape == (32, 80, 300)


def _test_pointwise_feed_forward():
    ffn = PointwiseFeedForward(d_model=300, d_ff=1024)
    x = tf.random.uniform((32, 80, 300))
    o = ffn(x)
    assert o.shape == (32, 80, 300)


def _test_encoder_layer():
    encoder_layer = EncoderLayer(n_heads=6, d_model=300, d_ff=768)
    x = tf.random.uniform((32, 80, 300))
    o = encoder_layer(x, None, False)
    assert o.shape == (32, 80, 300)


def _test_decoder_layer():
    encoder_layer = EncoderLayer(n_heads=6, d_model=300, d_ff=768)
    decoder_layer = DecoderLayer(n_heads=6, d_model=300, d_ff=768)
    x = tf.random.uniform((32, 80, 300))
    enc = encoder_layer(x)
    o = decoder_layer(x, enc)
    assert o.shape == (32, 80, 300)


def _test_encoder():
    encoder = Encoder(n_heads=6, d_model=300, num_layers=2, vocab_size=10000, max_seq_len=256, d_ff=768)
    x = tf.random.uniform((32, 80))
    o = encoder(x)
    # print(encoder.summary())
    assert o.shape == (32, 80, 300)


def _test_decoder():
    encoder = Encoder(n_heads=6, d_model=300, num_layers=2, vocab_size=10000, max_seq_len=256, d_ff=768)
    decoder = Decoder(n_heads=6, d_model=300, num_layers=2, vocab_size=10000, max_seq_len=256, d_ff=768)
    source = tf.random.uniform((32, 80))
    target = tf.random.uniform((32, 80))
    enc = encoder(source)
    o = decoder(target, enc)
    assert o.shape == (32, 80, 300)


def _test_transformer():
    model = Transformer(6, 300, 12, 768, 10000, 8000, 256)
    source = tf.random.uniform((32, 80))
    target = tf.random.uniform((32, 80))
    o = model(source, target, None, None, None, False)
    # print(model.summary())
    assert o.shape == (32, 80, 8000)


if __name__ == '__main__':
    _test_embedding()
    _test_pe()
    _test_attention()
    _test_multihead_attention()
    _test_pointwise_feed_forward()
    _test_encoder_layer()
    _test_decoder_layer()
    _test_encoder()
    _test_decoder()
    _test_transformer()

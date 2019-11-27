import numpy as np
import tensorflow as tf


__all__ = ['approx_ndcg_loss']


def _approx_ranks(logits, alpha=10.):
    """Compute approximated ranks.

    logits is of shape (# queries, # documents)

    rank_i = 1 + \sum_{j \neq i} I(s_j > s_i)
    That is: the more logits in the list that are larger than item_i's logit, the lower the rank of item_i is.

    An approximation of the indicator function is proposed by the referenced paper using generalized sigmoid:

    I(s_j > s_i) \approx 1 / (1 + exp(-\alpha * (s_j - s_i)))

    reference: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2008-164.pdf
    """
    N = tf.shape(logits)[1]
    i = tf.tile(tf.expand_dims(logits, 2), [1, 1, N])
    j = tf.tile(tf.expand_dims(logits, 1), [1, N, 1])
    pairs = tf.sigmoid(alpha * (j - i))
    ranks = tf.reduce_sum(pairs, axis=-1) + .5
    return ranks


def _gain_function(labels, base=2.):
    return tf.pow(base, tf.cast(labels, tf.float32)) - 1


def _discount_function(rank):
    return 1. / tf.math.log1p(tf.cast(rank, tf.float32))


def _sort_scores(scores, N=100):
    """Sort a 2D tensor row by row."""
    def _to_nd_indices(indices):
        batch_ids = tf.ones_like(indices) * tf.expand_dims(tf.range(tf.shape(indices)[0]), 1)
        return tf.stack([batch_ids, indices], axis=-1)

    scores = tf.cast(scores, tf.float32)
    N = tf.minimum(tf.shape(scores)[1], N)
    _, indices = tf.math.top_k(scores, N, sorted=True)
    nd_indices = _to_nd_indices(indices)
    return tf.gather_nd(scores, nd_indices)


def _inverse_max_dcg(labels, N=4):
    ideal_sorted_labels = _sort_scores(labels, N=N)
    rank = tf.cast(tf.range(tf.shape(ideal_sorted_labels)[1]) + 1, tf.float32)
    gain = _gain_function(ideal_sorted_labels)
    discount = _discount_function(rank)
    discounted_gain = gain * discount
    discounted_gain = tf.reduce_sum(discounted_gain, axis=1, keepdims=True)
    discounted_gain = tf.where(
        tf.greater(discounted_gain, 0.),
        1. / discounted_gain,
        tf.zeros_like(discounted_gain)
    )
    return discounted_gain


def approx_ndcg_loss(labels, logits, alpha=10., epsilon=1e-10, dtype=tf.float32, reduction='mean'):
    """Approximation of NDCG loss.

    labels and logits are of shape, # queries x query size
    The queries needs to be padded to equal length.
    """
    labels = tf.cast(labels, dtype)
    logits = tf.cast(logits, dtype)
    label_sum = tf.reduce_sum(labels, axis=1, keepdims=True)
    nonzero_mask = tf.greater(label_sum, 0.)
    labels = tf.where(nonzero_mask, labels, epsilon * tf.ones_like(labels, dtype))
    gains = _gain_function(labels)
    ranks = _approx_ranks(logits, alpha)
    discounts = _discount_function(ranks)
    dcg = tf.reduce_sum(gains * discounts, axis=-1)
    loss = -dcg * tf.squeeze(_inverse_max_dcg(labels))
    if reduction == 'mean':
        loss = tf.reduce_mean(loss)
    elif reduction == 'sum':
        loss = tf.reduce_sum(loss)
    return loss


def _test_approx_ranks():
    x = tf.convert_to_tensor(tf.cast(np.array([[3, 4, 5], [2, 1, 3]]), tf.float32))
    r = _approx_ranks(x)
    assert np.isclose(r, np.array([[3, 2, 1], [2, 3, 1]]), rtol=1e-3).all()


def _test_rank_loss_value():
    labels = tf.convert_to_tensor(np.array([[1., 2., 3.]]))
    logits1 = tf.convert_to_tensor(np.array([[3.5, 4.5, 6.0]]))
    logits2 = tf.convert_to_tensor(np.array([[3.5, 2.5, 6.0]]))
    logits3 = tf.convert_to_tensor(np.array([[6.0, 4.5, 1.0]]))
    loss1 = approx_ndcg_loss(labels, logits1, reduction='mean').numpy()
    loss2 = approx_ndcg_loss(labels, logits2, reduction='mean').numpy()
    loss3 = approx_ndcg_loss(labels, logits3, reduction='mean').numpy()
    assert loss3 > loss2 > loss1


def _test_rank_loss_batch():
    labels = tf.convert_to_tensor(np.array([[1., 2., 3.], [3., 2., 1.]]))
    logits1 = tf.convert_to_tensor(np.array([[4., 6., 8.], [5., 4., 10.]]))
    logits2 = tf.convert_to_tensor(np.array([[4., 6., 8.], [5., 4., 3.]]))
    loss1 = approx_ndcg_loss(labels, logits1, reduction='mean').numpy()
    loss2 = approx_ndcg_loss(labels, logits2, reduction='mean').numpy()
    assert loss1 > loss2


if __name__ == '__main__':
    _test_approx_ranks()
    _test_rank_loss_value()
    _test_rank_loss_batch()

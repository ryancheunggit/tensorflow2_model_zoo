import numpy as np


class DCG(object):
    """Discounted cumulative gain.

    DCG measures the quality of a ranking task.

    DCG_p = \Sigma_{i = 1}^{p} \frac{rel_i}{log_2(i + 1)}
    where rel_i is the graded relevance of the result at position i.
          log_2(i + 1) is the discount factor
    this formula correspond to gain_type == 'identity'

    An alternative formulation of DCG places stronger emphasis on retrieving relevant documents.
    DCG_p = \Sigma_{i=1}^{p} \frac{2^{rel_i} - 1}{log_2(i + 1}
    this formula correspond to gain_type == 'exp2'
    """
    def get_discount(self):
        return self.__discount

    def set_discount(self, n):
        self.__discount = np.array([np.log2(x + 1) for x in range(1, n + 1)])

    def del_discount(self):
        del self.__discount

    discount = property(get_discount, set_discount, del_discount, 'cached discount values')

    def __init__(self, k=100, gain_type='exp2'):
        assert gain_type in ['identity', 'exp2']
        self.k = k
        self.set_discount(k)
        self.gain_type = gain_type

    def evaluate(self, rels):
        rels = np.array(rels)[:self.k]
        if self.gain_type == 'identity':
            gain = rels
        else:
            gain = np.power(2.0, rels) - 1.
        discount = self.get_discount()[:min(self.k, len(gain))]
        return np.sum(np.divide(gain, discount))


class NDCG(DCG):
    def __init__(self, k=100, gain_type='exp2'):
        super(NDCG, self).__init__(k, gain_type)

    def evaluate(self, rels):
        dcg = super(NDCG, self).evaluate(rels)
        max_dcg = self.get_max_DCG(rels)
        return dcg / max_dcg

    def get_max_DCG(self, rels):
        return super(NDCG, self).evaluate(np.sort(np.array(rels))[::-1])



def _test_DCG():
    o = DCG(k=4, gain_type='identity')
    assert all(o.discount == np.array(np.log2([2, 3, 4, 5])))
    assert np.isclose(o.evaluate([3, 2, 1, 0]), 4.761859)
    assert np.isclose(o.evaluate([0, 1, 2, 3]), 2.922959)
    o = DCG(k=4, gain_type='exp2')
    assert all(o.discount == np.array(np.log2([2, 3, 4, 5])))
    assert np.isclose(o.evaluate([3, 2, 1, 0]), 9.392789)
    assert np.isclose(o.evaluate([0, 1, 2, 3]), 5.145665)


def _test_NDCG():
    m = NDCG(k=4, gain_type='identity')
    assert m.evaluate([3, 2, 1, 0]) == 1.
    assert np.isclose(m.evaluate([0, 1, 2, 3]), 0.613827)
    m = NDCG(k=4, gain_type='exp2')
    assert m.evaluate([3, 2, 1, 0]) == 1.
    assert np.isclose(m.evaluate([0, 1, 2, 3]), 0.547831)


if __name__ == '__main__':
    _test_DCG()
    _test_NDCG()

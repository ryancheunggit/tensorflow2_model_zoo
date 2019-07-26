import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import nn
from tensorflow.python.framework import ops


class ReLU6(keras.Model):
    '''Rectified Linear 6, min(max(x, 0), 6).

    It is said to be more robust when used with low-precision.
    '''
    def __init__(self, name):
        super(ReLU6, self).__init__(name=name)

    def call(self, x):
        return nn.relu6(x)


# Swish - https://arxiv.org/abs/1710.05941.
def swish(features, beta=1, name=None):
    with ops.name_scope(name, 'Swish', [features, beta]) as name:
        features = ops.convert_to_tensor(features, name='features')
        return features * tf.sigmoid(beta * features)


class Swish(keras.Model):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def call(self, x):
        return swish(x, self.beta)


# Hard Sigmoid - https://arxiv.org/abs/1905.02244.
def hsigmoid(features, name=None):
    with ops.name_scope(name, 'HSigmoid', [features]) as name:
        features = ops.convert_to_tensor(features, name='features')
        return nn.relu6(features + 3.0) / 6.0


class HSigmoid(keras.Model):
    def call(self, x):
        return hsigmoid(x)


# H-Swish - https://arxiv.org/abs/1905.02244.
def hswish(features, name=None):
    with ops.name_scope(name, 'HSwish', [features]) as name:
        features = ops.convert_to_tensor(features, name='features')
        return features * nn.relu6(features + 3.0) / 6.0


class HSwish(keras.Model):
    def call(self, x):
        return hswish(x)


def _test_swish():
    sigmoid = lambda x: 1. / (1. + np.exp(-x))
    x = tf.random.uniform((32, 1))
    beta = 1
    from_numpy = x.numpy() * sigmoid(x.numpy() * beta)
    from_tf_function = swish(x, beta)
    from_tf_class = Swish(beta=beta)(x)
    assert np.allclose(from_numpy, from_tf_function)
    assert np.allclose(from_numpy, from_tf_class)


def _test_hsigmoid():
    relu6 = lambda x: np.minimum(np.maximum(x, 0), 6)
    x = tf.random.uniform((32, 1))
    from_numpy = relu6(x.numpy() + 3) / 6
    from_tf_function = hsigmoid(x)
    from_tf_class = HSigmoid()(x)
    assert np.allclose(from_numpy, from_tf_function)
    assert np.allclose(from_numpy, from_tf_class)


def _test_hswish():
    relu6 = lambda x: np.minimum(np.maximum(x, 0), 6)
    x = tf.random.uniform((32, 1))
    from_numpy = x.numpy() * relu6(x.numpy() + 3) / 6
    from_tf_function = hswish(x)
    from_tf_class = HSwish()(x)
    assert np.allclose(from_numpy, from_tf_function)
    assert np.allclose(from_numpy, from_tf_class)


def _test_all():
    _test_swish()
    _test_hsigmoid()
    _test_hswish()


if __name__ == '__main__':
    _test_all()

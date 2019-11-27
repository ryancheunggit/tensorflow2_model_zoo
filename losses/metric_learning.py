import math
import tensorflow as tf


class AdditiveAngularMarginLoss(tf.keras.Model):
    """Implementation of Additive Angular Margin Loss.

    reference: https://arxiv.org/abs/1801.07698
    """
    def __init__(self, s=16, m=.3, num_classes=10):
        super(AdditiveAngularMarginLoss, self).__init__()
        self.s = s
        self.m = m
        self.num_classes = num_classes
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def call(self, labels, cos_theta):
        sin_theta = tf.sqrt(1.0 - tf.square(cos_theta))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        phi = tf.where(cos_theta > self.th, phi, cos_theta - self.mm)
        one_hot = tf.one_hot(labels, depth=self.num_classes, dtype='float32')
        out = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        out = out * self.s
        return self.criterion(one_hot, out)

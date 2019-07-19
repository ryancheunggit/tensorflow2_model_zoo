"""Example program on metric learning with ArcFace."""
import argparse
import os
import math
import numpy as np
import tensorflow as tf
from datetime import datetime
from convnets.common import Conv2d, MaxPool2d
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.layers import BatchNormalization, ReLU, GlobalMaxPooling2D, Dense


BATCH_SIZE = 32
LEARNING_RATE = 1e-3

if not os.path.exists('models/mnist_mlp_metric_learning/'):
    os.mkdir('models/mnist_mlp_metric_learning/')
MODEL_FILE = 'models/mnist_mlp_metric_learning/model'


class ConvBlock(tf.keras.Model):
    """Conv2d -> BatchNorm -> ReLU -> MaxPool"""
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding, data_format, name='conv_block'):
        super(ConvBlock, self).__init__()
        self.conv2d = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             strides=strides, padding=padding, use_bias=False, data_format=data_format,
                             name=name + '/conv')
        self.batchnorm = BatchNormalization(axis=-1 if data_format == 'channels_last' else 1, name=name + '/bn')
        self.activation = ReLU(name=name + '/activ')
        self.pool = MaxPool2d(pool_size=2, strides=2, padding=0, ceil_mode=False, data_format=data_format,
                              name=name + 'pool')

    def call(self, x, training=False):
        return self.pool(self.activation(self.batchnorm(self.conv2d(x), training=training)))


class CosineLinear(tf.keras.Model):
    """Linear layer without bias, both the weight vector and feature vector are l2 normalized.

    W.T dot X = ||W|| * ||X|| * cos(theta).
    """
    def __init__(self, num_features, num_classes=10):
        super(CosineLinear, self).__init__()
        self.weight = tf.random.uniform((num_features, num_classes), dtype='float32')

    def call(self, x):
        cos_theta = tf.matmul(tf.math.l2_normalize(x), tf.math.l2_normalize(self.weight))
        return cos_theta


class AdditiveAngularMarginLoss(tf.keras.Model):
    """Implementation of Additive Angular Margin Loss.

    reference: https://arxiv.org/abs/1801.07698
    """
    def __init__(self, s=16, m=.3, num_classes=10):
        super(AdditiveAngularMarginLoss, self).__init__()
        self.s = s
        self.m = m
        self.num_classes = 10
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def call(self, labels, cos_theta):
        sin_theta = tf.pow(1.0 - tf.pow(cos_theta, 2), .5)
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        phi = tf.where(cos_theta > self.th, phi, cos_theta - self.mm)
        one_hot = tf.one_hot(labels, depth=self.num_classes, dtype='float32')
        out = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        out = out * self.s
        return self.criterion(one_hot, out)


class ConvNet(tf.keras.Model):
    def __init__(self, n_hidden=128, num_classes=10, last_linear='cosine'):
        super(ConvNet, self).__init__()
        self.features = tf.keras.Sequential([
            ConvBlock(in_channels=1, out_channels=32, kernel_size=3, strides=1, padding=0,
                      data_format='channels_last', name='features/conv1'),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, strides=1, padding=0,
                      data_format='channels_last', name='features/conv2'),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=3, strides=1, padding=0,
                      data_format='channels_last', name='features/conv2'),
            GlobalMaxPooling2D(data_format='channels_last', name='pool'),
            Dense(units=n_hidden, name='features/fc1'),
            ReLU(name='features/relu')
        ])
        if last_linear == 'cosine':
            self.last_linear = CosineLinear(num_features=n_hidden, num_classes=num_classes)
        else:
            self.last_linear = Dense(num_classes, use_bias=False)

    def call(self, x, training=False):
        features = self.features(x, training=training)
        out = self.last_linear(features)
        return out

    def hidden(self, x, training=False):
        return self.features(x, training=training)


def main(verbose=0):
    verbose = verbose
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.0
    x_valid = x_valid.reshape(10000, 28, 28, 1).astype('float32') / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(BATCH_SIZE)

    def train_model(model, criterion, optimizer, max_epochs, min_acc=.98):
        train_loss = tf.keras.metrics.Mean()
        test_loss = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        @tf.function
        def train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                out = model(x_batch, training=True)
                loss = criterion(y_batch, out)
            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            train_loss(loss)
            train_accuracy(y_batch, out)

        @tf.function
        def valid_step(x_batch, y_batch):
            out = model(x_batch, training=False)
            loss = criterion(y_batch, out)
            test_loss(loss)
            test_accuracy(y_batch, out)

        # training loop
        for epoch in range(max_epochs):
            t0 = datetime.now()
            # train
            for idx, (x_batch, y_batch) in enumerate(train_dataset):
                train_step(x_batch, y_batch)

            # validate
            for idx, (x_batch, y_batch) in enumerate(valid_dataset):
                valid_step(x_batch, y_batch)

            message_template = 'epoch {:>3} time {} sec / epoch train loss {:.4f} acc {:4.2f}% ' + \
                               'test loss {:.4f} acc {:4.2f}%'
            t1 = datetime.now()
            if verbose:
                print(message_template.format(
                    epoch + 1, (t1 - t0).seconds,
                    train_loss.result(), train_accuracy.result() * 100,
                    test_loss.result(), test_accuracy.result() * 100
                ))
            if float(test_accuracy.result()) > min_acc:
                print('terminated for that minimal test accuracy reached')
                break

        return model

    def plot_feature(model, type='hidden', fpath='plots/tsne.png'):
        @tf.function
        def get_hidden(model, x_batch):
            return model.hidden(x_batch)

        @tf.function
        def get_logits(model, x_batch):
            return model(x_batch)

        features = []
        for _, (x_batch, _) in enumerate(valid_dataset):
            x_out = get_hidden(model, x_batch) if type == 'hidden' else get_logits(model, x_batch)
            features.append(x_out)
        features = np.vstack(features)
        features_embedded = TSNE(n_components=2).fit_transform(features)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(features_embedded[:, 0], features_embedded[:, 1], c=y_valid, alpha=.33, label=y_valid)
        # plt.show()
        fig.savefig(fpath)

    # config model
    model = ConvNet(n_hidden=128, num_classes=10, last_linear='cosine')
    criterion = AdditiveAngularMarginLoss(s=30, m=.3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model = train_model(model, criterion, optimizer, max_epochs=100)
    plot_feature(model, 'hidden', 'plots/mnist_metric_learning_hidden_tsne.png')
    plot_feature(model, 'logits', 'plots/mnist_metric_learning_logits_tsne.png')

    model = ConvNet(n_hidden=128, num_classes=10, last_linear='linear')
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model = train_model(model, criterion, optimizer, max_epochs=100)
    plot_feature(model, 'hidden', 'plots/mnist_linear_cce_hidden_tsne.png')
    plot_feature(model, 'logits', 'plots/mnist_linear_cce_logits_tsne.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters for program')
    parser.add_argument('--gpu', default='', help='gpu device id expose to program, default is cpu only.')
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args.verbose)

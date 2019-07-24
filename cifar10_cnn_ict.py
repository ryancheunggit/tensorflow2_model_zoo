"""Example program training/inference with a CNN image classifier trained on cifar10 with tensorflow 2.0."""
import argparse
import cv2
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from itertools import cycle
from tensorflow import keras
from tensorflow.keras import layers

# This is a modification to the cifar10_cnn_mixup.py
# We are trying 'Interpolation Consistency Training for Semi-Supervised Learning' https://arxiv.org/pdf/1903.03825.pdf


BATCH_SIZE = 32
NUM_CLASS = 10
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
MIXUP_ALPHA = .5
EMA_DECAY = .999
W_MAX = 50

if not os.path.exists('models/cifar10_cnn_ict/'):
    os.mkdir('models/cifar10_cnn_ict/')
MODEL_FILE = 'models/cifar10_cnn_ict/model'
CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class ConvBlock(keras.Model):
    """Convolution block: (Conv - BN - ReLu) * 2 -> Avg Pooling."""
    def __init__(self, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters=out_channels, kernel_size=(3, 3), use_bias=False)  # BN absorbes bias
        self.conv2 = layers.Conv2D(filters=out_channels, kernel_size=(3, 3), use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.pooling = layers.AveragePooling2D(pool_size=(2, 2))

    def call(self, x, training=True):
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        x = self.pooling(x)
        return x


class MaxAvgPool(keras.Model):
    """Global pooling."""
    def __init__(self):
        super(MaxAvgPool, self).__init__()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.avg_pool = layers.GlobalAveragePooling2D()

    def call(self, x):
        x = 0.5 * (self.max_pool(x) + self.avg_pool(x))
        return x


class Classifier(keras.Model):
    """A simple classifier with one hidden layer."""
    def __init__(self, num_class):
        super(Classifier, self).__init__()
        self.hidden = layers.Dense(units=32)
        self.classifier = layers.Dense(units=num_class)

    def call(self, x, training=True):
        if training:
            x = tf.nn.relu(tf.nn.dropout(self.hidden(x), .2))
        else:
            x = tf.nn.relu(self.hidden(x))
        x = tf.nn.softmax(self.classifier(x))
        return x


class CNN(keras.Model):
    """Convolutional Network."""
    def __init__(self, sizes=(32, 64), num_class=NUM_CLASS):
        super(CNN, self).__init__()
        self.feature_extraction = keras.Sequential([
            ConvBlock(out_channels=size) for size in sizes
        ])
        self.feature = MaxAvgPool()
        self.classifier = Classifier(num_class=NUM_CLASS)

    def call(self, x, training=True):
        x = self.feature_extraction(x, training=training)
        x = self.feature(x)
        x = self.classifier(x, training=training)
        return x


def _copy_model_weights(from_model, to_model):
    """Copy weights from a model to a model."""
    for from_val, to_val in zip(from_model.variables, to_model.variables):
        to_val.assign(from_val)


def _ema_model_weights(from_model, to_model, decay=EMA_DECAY):
    """Exponential Moving Average of model's weights were stored as ema_model."""
    for from_val, to_val in zip(from_model.variables, to_model.variables):
        to_val.assign(decay * to_val + (1 - decay) * from_val)


def train(verbose=0):
    """Train the model."""
    # load dataset
    cifar10 = keras.datasets.cifar10
    (x_train, y_train), (x_valid, y_valid) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_valid = x_valid.astype('float32') / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)
    train_u_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(2 * BATCH_SIZE)
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(BATCH_SIZE)

    # config model
    model = CNN()
    ema_model = CNN()  # the teacher model

    # initialize and sync model weights
    # TODO: how do i initialize weighte with out passing an actual batch?
    init_batch, _ = next(iter(train_dataset.shuffle(32).take(1)))
    _, _ = model(init_batch, training=False), ema_model(init_batch, training=False)
    _copy_model_weights(ema_model, model)

    data_criterion = keras.losses.SparseCategoricalCrossentropy()
    consistency_criterion = keras.losses.MeanSquaredError()

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    train_data_loss = keras.metrics.Mean()
    train_consistency_loss = keras.metrics.Mean()
    test_loss = keras.metrics.Mean()
    train_accuracy = keras.metrics.SparseCategoricalAccuracy()
    test_accuracy = keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def mixup(u1, u2, alpha=MIXUP_ALPHA):
        """Mixup unlabeled data, also use teacher model to guess label."""
        lambd = np.random.beta(alpha, alpha)
        q1 = ema_model(u1, training=False)
        q2 = ema_model(u2, training=False)
        u_mixed = lambd * u1 + (1 - lambd) * u2
        q_guessed = lambd * q1 + (1 - lambd) * q2
        return u_mixed, q_guessed

    @tf.function
    def train_step(x_batch, y_batch, u_batch_1, u_batch_2, w):
        u_mixed, q_guessed = mixup(u_batch_1, u_batch_2)

        with tf.GradientTape() as tape:
            u_out = model(u_mixed, training=False)
            x_out = model(x_batch, training=True)
            data_loss = data_criterion(y_batch, x_out)
            consistency_loss = w * consistency_criterion(q_guessed, u_out)
            loss = data_loss + consistency_loss

        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        _ema_model_weights(model, ema_model)
        train_data_loss(data_loss)
        train_consistency_loss(consistency_loss)
        train_accuracy(y_batch, x_out)

    @tf.function
    def valid_step(x_batch, y_batch):
        out = model(x_batch, training=False)
        loss = data_criterion(y_batch, out)
        test_loss(loss)
        test_accuracy(y_batch, out)

    # training loop
    for epoch in range(NUM_EPOCHS):
        t0 = datetime.now()
        # calculate ict loss weights, this weight peaks at W_MAX in sigmoid fashion from epoch 0 to 1/4 total epochs
        w = min(W_MAX, 50 / (1 + np.exp(NUM_EPOCHS / 8 - epoch)))
        # train
        for idx, ((x_batch, y_batch), (u_batch, _)) in enumerate(zip(train_dataset, cycle(train_u_dataset))):
            if u_batch.shape[0] != x_batch.shape[0] * 2:
                continue
            u_batch_1 = u_batch[:BATCH_SIZE]
            u_batch_2 = u_batch[BATCH_SIZE:]
            train_step(x_batch, y_batch, u_batch_1, u_batch_2, w)

        # validate
        for idx, (x_batch, y_batch) in enumerate(valid_dataset):
            valid_step(x_batch, y_batch)

        message_template = 'epoch {:>3} time {} sec / epoch train cce {:.4f} ict {:.4f} acc {:4.2f}% test cce {:.4f} acc {:4.2f}%'
        t1 = datetime.now()
        if verbose:
            print(message_template.format(
                epoch + 1, (t1 - t0).seconds,
                train_data_loss.result(), train_consistency_loss.result(), train_accuracy.result() * 100,
                test_loss.result(), test_accuracy.result() * 100
            ))

    model.save_weights(MODEL_FILE, save_format='tf')


def inference(filepath):
    """Reconstruct the model, load weights and run inference on a given picture."""
    model = CNN()
    model.load_weights(MODEL_FILE)
    image = cv2.imread(filepath).astype('float32') / 255
    # Somehow model.predict is throwing error for me, I am doing manual convert here
    probs = model(np.expand_dims(image, 0)).numpy()
    print('it is a: {} with probability {:4.2f}%'.format(CIFAR10_LABELS[probs.argmax()], 100 * probs.max()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters for program')
    parser.add_argument('procedure', choices=['train', 'inference'],
                        help='Whether to train a new model or use trained model to inference.')
    parser.add_argument('--image_path', default=None, help='Path to jpeg image file to predict on.')
    parser.add_argument('--gpu', default='', help='gpu device id expose to program, default is cpu only.')
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.procedure == 'train':
        train(args.verbose)
    else:
        assert os.path.exists(MODEL_FILE + '.index'), 'model not found, train a model before calling inference.'
        assert os.path.exists(args.image_path), 'can not find image file.'
        inference(args.image_path)

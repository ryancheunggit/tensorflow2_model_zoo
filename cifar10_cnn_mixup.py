"""Example program training/inference with a CNN image classifier trained on cifar10 with tensorflow 2.0."""
import argparse
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

# This is a modification to the cifar10_cnn.py
# We are adding mixup to the training
# With mixup, it is encourging the model to have near linear behavior between categories.
# Note that I am using tf.gather, because unlike pytorch, numpy style sclicing seems won't work with tf Tensors.


BATCH_SIZE = 32
NUM_CLASS = 10
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
MIXUP_ALPHA = .2

if not os.path.exists('models/cifar10_cnn_mixup/'):
    os.mkdir('models/cifar10_cnn_mixup/')
MODEL_FILE = 'models/cifar10_cnn_mixup/model'
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

    def call(self, x):
        x = tf.nn.relu(self.bn1(self.conv1(x)))
        x = tf.nn.relu(self.bn2(self.conv2(x)))
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

    def call(self, x):
        x = tf.nn.relu(tf.nn.dropout(self.hidden(x), .2))
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

    def call(self, x):
        x = self.feature_extraction(x)
        x = self.feature(x)
        x = self.classifier(x)
        return x


def train(verbose=0):
    """Train the model."""
    # load dataset
    cifar10 = keras.datasets.cifar10
    (x_train, y_train), (x_valid, y_valid) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_valid = x_valid.astype('float32') / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(BATCH_SIZE)

    # config model
    model = CNN()
    criterion = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    train_loss = keras.metrics.Mean()
    test_loss = keras.metrics.Mean()
    train_accuracy = keras.metrics.SparseCategoricalAccuracy()
    test_accuracy = keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def mixup_data(x, y, alpha=MIXUP_ALPHA):
        """Mixup data in the batch."""
        lambd = np.random.beta(alpha, alpha)
        indices = np.random.permutation(range(x.shape[0]))
        x_alt = tf.gather(x, indices)
        y_alt = tf.gather(y, indices)
        x_mixed = lambd * x + (1 - lambd) * x_alt
        return x_mixed, y_alt, lambd

    @tf.function
    def train_step(x_batch, y_batch, y_alt, lambd):
        with tf.GradientTape() as tape:
            out = model(x_batch)
            loss = lambd * criterion(y_batch, out) + (1 - lambd) * criterion(y_alt, out)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        train_loss(loss)
        train_accuracy(y_batch, out)

    @tf.function
    def valid_step(x_batch, y_batch):
        out = model(x_batch)
        loss = criterion(y_batch, out)
        test_loss(loss)
        test_accuracy(y_batch, out)

    # training loop
    for epoch in range(NUM_EPOCHS):
        t0 = datetime.now()
        # train
        for idx, (x_batch, y_batch) in enumerate(train_dataset):
            x_mixed, y_alt, lambd = mixup_data(x_batch, y_batch)
            train_step(x_mixed, y_batch, y_alt, lambd)

        # validate
        for idx, (x_batch, y_batch) in enumerate(valid_dataset):
            valid_step(x_batch, y_batch)

        message_template = 'epoch {:>3} time {} sec / epoch train cce {:.4f} acc {:4.2f}% test cce {:.4f} acc {:4.2f}%'
        t1 = datetime.now()
        if verbose:
            print(message_template.format(
                epoch + 1, (t1 - t0).seconds,
                train_loss.result(), train_accuracy.result() * 100,
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

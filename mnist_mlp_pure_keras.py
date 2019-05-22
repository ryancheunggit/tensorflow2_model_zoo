"""Example program training/inference on digit recognition problem with tensorflow 2.0."""
import argparse
import cv2
import os
import tensorflow as tf
from tensorflow import keras

# This is a modification to the mnist_mlp_keras_sequential.py
# We are using keras.model.fit to replace the custom train/validation process
# Keras wants to enforce us to shuffle the dataset, but it seems to add quite a bit overhead.
# Using model.fit is actually noticeably slower than custom training loop
# model save/load are still broken even now we are pure keras in model struct, compile, train and save/load
# found a issue https://github.com/keras-team/keras/issues/10417
# I don't see any advantage using model.fit, we lose customization and loss speed too.


BATCH_SIZE = 32
NUM_CLASS = 10
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
if not os.path.exists('models/mnist_mlp_keras_sequential/'):
    os.mkdir('models/mnist_mlp_keras_sequential/')
MODEL_FILE = 'models/mnist_mlp_keras_sequential/model.h5'


def MLP():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(784,)))
    model.add(keras.layers.Dense(units=128))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Dense(units=32))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation='relu'))
    model.add(keras.layers.Dense(units=NUM_CLASS))
    model.add(keras.layers.Dropout(rate=.1))
    model.add(keras.layers.Activation(activation='softmax'))
    return model


def train(verbose=0):
    """Train the model."""
    # load dataset
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
    x_valid = x_valid.reshape(10000, 784).astype('float32') / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1000).batch(BATCH_SIZE)
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(BATCH_SIZE)

    # config model
    model = MLP()
    criterion = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=criterion, metrics=['sparse_categorical_accuracy'])

    # training
    model.fit(x=train_dataset, epochs=NUM_EPOCHS, validation_data=valid_dataset)

    model.save(MODEL_FILE)  # still, save this way, won't be able to load


def inference(filepath):
    """Reconstruct the model, load whole model directly and run inference on a given picture."""
    try:
        model = keras.models.load_model(MODEL_FILE)
        # model = keras.experimental.load_from_saved_model(MODEL_FILE)
        image = cv2.imread(filepath, 0).reshape(1, 784).astype('float32') / 255
        probs = model.predict(image)
        print('it is a: {} with probability {:4.2f}%'.format(probs.argmax(), 100 * probs.max()))
    except:
        print('inference failed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters for program')
    parser.add_argument('procedure', choices=['train', 'inference'],
                        help='Whether to train a new model or use trained model to inference.')
    parser.add_argument('--image_path', default=None, help='Path to jpeg image file to predict on.')
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()
    if args.procedure == 'train':
        train(args.verbose)
    else:
        assert os.path.exists(MODEL_FILE), 'model not found, train a model before calling inference.'
        assert os.path.exists(args.image_path), 'can not find image file.'
        inference(args.image_path)

import argparse
import cv2
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from resnet import resnet50

parser = argparse.ArgumentParser(description='cnn models benchmark runnerr')
parser.add_argument('model', type=str)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--patience', type=int, default = 5)
parser.add_argument('--earlystop', type=int, default = 10)
parser.add_argument('--max_epoch', type=int, default = 200)
parser.add_argument('--steps_per_epoch', type=int, default=10000)
parser.add_argument('--valid_steps', type=int, default=500)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

MODEL_PATH = '/home/renzhang/projects/tf2_zoo/models/imagenet_models/'
LOG_PATH = '/home/renzhang/projects/tf2_zoo/models/imagenet_models/'
IMAGE_RES = (224, 224)
MEANS = [.485, .456, .406]
STDS = [.229, .224, .225]
AUTOTUNE = tf.data.experimental.AUTOTUNE


def _get_model(args):
    if args.model.lower() == 'resnet50':
        model = resnet50()
    return model


def _process_image(image, label, augment=False):
    image = tf.cast(image, tf.float32)
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_contrast(image, .8, 1.2)
        image = tf.image.random_brightness(image, .2)
        image = tf.image.random_hue(image, .2)
        image = tf.image.random_saturation(image, .8, 1.2)
    image /= 255.0
    image = tf.image.resize(image, IMAGE_RES)
    image -= MEANS
    image /= STDS
    return image, label


def main():
    data = tfds.load('imagenet2012', as_supervised=True)
    train_dataset = data['train'].\
            map(partial(_process_image, augment=True), num_parallel_calls=AUTOTUNE).\
            shuffle(1024).\
            repeat().\
            batch(args.batch_size)
    valid_dataset = data['validation'].\
            map(partial(_process_image, augment=False), num_parallel_calls=AUTOTUNE).\
            shuffle(1024).\
            repeat().\
            batch(args.batch_size)
    best_score = 0
    best_epoch = 0
    lr = args.learning_rate
    patience = args.patience
    earlystop = args.earlystop

    model = _get_model(args)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = criterion(y, logits)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return logits, y, loss

    @tf.function
    def valid_step(x, y):
        logits = model(x, training=False)
        loss = criterion(y, logits)
        return logits, y, loss

    for epoch in range(1, args.max_epoch + 1):
        train_loss = tf.keras.metrics.Mean()
        train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

        for step, (x, y) in enumerate(train_dataset):
            logits, y, loss = train_step(x, y)
            train_loss(loss)
            train_acc(y, logits)
            print('\repoch {:>4} step {:>6} train loss {:.4f} - accuracy {:.4f}'.format(
                epoch, step, train_loss.result(), train_acc.result()), end='', flush=True)
            if step > args.steps_per_epoch > 0:
                break
        print('\r\n', end='', flush=False)

        valid_loss = tf.keras.metrics.Mean()
        valid_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        for step, (x, y) in enumerate(valid_dataset):
            logits, y, loss = valid_step(x, y)
            valid_loss(loss)
            valid_acc(y, logits)
            if step > args.valid_steps > 0:
                break
        valid_message = 'epoch {:>4} validation loss {:.4f} - accuracy {:.4f}'.format(
            epoch, valid_loss.result(), valid_acc.result())
        print(valid_message)
        with open(os.path.join(LOG_PATH, '{}.txt'.format(args.model)), 'a+') as f:
            f.write(args.model + ' ' + valid_message + '\n')

        score = valid_acc.result()
        if score > best_score:
            best_score = score
            best_epoch = epoch
            model.save_weights(os.path.join(MODEL_PATH, args.model), save_format='tf')
        else:
            if epoch - best_epoch > patience:
                lr *= .1
                optimizer.lr.assign(lr)
                print('--learning rate reduced to {}'.format(lr))
            if epoch - best_epoch > earlystop:
                print('--early stopping triggered, best validation accuracy {} at epoch {}'.format(
                    best_score, best_epoch))


if __name__ == '__main__':
    main()

import argparse
import gc
import math
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from afm import AttentionalFactorizationMachine
from collections import defaultdict
from fm import FactorizationMachine
from ffm import FieldAwareFactorizationMachine
from fnn import FMNeuralNetwork
from functools import partial
from sklearn.model_selection import train_test_split
from tqdm import tqdm


parser = argparse.ArgumentParser(description='fm models benchmark runner')
parser.add_argument('--model', type=str, default='FM')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=9000)  # It's Over 9000!
parser.add_argument('--factor_dim', type=int, default=16)
parser.add_argument('--fnn_hidden', type=str, default='256,128,1')
parser.add_argument('--afm_attn_size',  type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--patience', type=int, default=1)
parser.add_argument('--earlystop', type=int, default=3)
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--steps_per_epoch', type=int, default=1000)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gc.enable()


DATA_PATH = '/home/renzhang/projects/tf2_zoo/data/criteo/'
TRAIN_DATA = os.path.join(DATA_PATH, 'train.txt')
TEST_DATA = os.path.join(DATA_PATH, 'test.txt')
TRAIN_FEATURES = os.path.join(DATA_PATH, 'train_features.npy')
TEST_FEATURES = os.path.join(DATA_PATH, 'test_features.npy')
TRAIN_TARGETS = os.path.join(DATA_PATH, 'train_targets.npy')
ENCODERS_CACHE = os.path.join(DATA_PATH, 'encoders.pkl')
LOG_FILE = os.path.join(DATA_PATH, 'log.txt')
MODEL_PATH = os.path.join(DATA_PATH, args.model)
MIN_FREQ = 10
COL_NAMES = [str(i) for i in range(1, 40)]
DTYPES = {str(i): 'str' for i in range(1, 40)}


def convert_num_features(val):
    if val == '':
        return 'NULL'
    val = int(val)
    if val > 2:
        return str(int(math.log(val) ** 2))
    else:
        return str(val - 2)


def create_feature_encoders(datafile, min_freq=MIN_FREQ):
    """go through dataset once and create feature encoders."""
    feature_counts = {i: defaultdict(int) for i in range(1, 40)}

    with open(datafile) as f:
        for line in tqdm(f):
            values = line.rstrip('\n').split('\t')
            if len(values) != 40:
                continue
            for i, val in enumerate(values[1: 14], 1):
                feature_counts[i][convert_num_features(val)] += 1
            for i, val in enumerate(values[14:], 14):
                feature_counts[i][val] += 1
    feature_encoders = {
        i: set({val for val, freq in val_freq.items() if freq >= min_freq})
        for i, val_freq in feature_counts.items()
    }
    feature_encoders = {
        i: {val: encoded_val for encoded_val, val in enumerate(vals, 1)}
        for i, vals in feature_encoders.items()
    }
    return feature_encoders


def load_feature_encoders(encoders_path=ENCODERS_CACHE):
    if not os.path.exists(encoders_path):
        feature_encoders = create_feature_encoders(TRAIN_DATA)
        pickle.dump(feature_encoders, open(encoders_path, 'wb'))
    else:
        feature_encoders = pickle.load(open(encoders_path, 'rb'))
    return feature_encoders


def encode_dataset(datapath, feature_encoders, is_testset=False):
    features, targets = [], []
    with open(datapath) as f:
        for line in tqdm(f, desc='encode dataset', leave=False):
            feature = []
            values = line.rstrip('\n').split('\t')
            if is_testset:
                values = [0] + values
            target = int(values[0])
            for i, val in enumerate(values[1: 14], 1):
                feature.append(feature_encoders[i].get(convert_num_features(val), 0))
            for i, val in enumerate(values[14:], 14):
                feature.append(feature_encoders[i].get(val, 0))
            features.append(feature)
            targets.append(target)
    features = np.vstack(features)
    targets = np.array(targets)
    return features, targets


def get_encoded_dataset(feature_encoders, get_testset=False):
    if not os.path.exists(TRAIN_FEATURES):
        train_features, train_targets = encode_dataset(TRAIN_DATA, feature_encoders, is_testset=False)
        np.save(TRAIN_FEATURES, train_features)
        np.save(TRAIN_TARGETS, train_targets)
    else:
        train_features = np.load(TRAIN_FEATURES)
        train_targets = np.load(TRAIN_TARGETS)

    if get_testset:
        if not os.path.exists(TEST_FEATURES):
            test_features, _ = encode_dataset(TEST_DATA, feature_encoders, is_testset=True)
            np.save(TEST_FEATURES, test_features)
        else:
            test_features = np.load(TEST_FEATURES)
        return train_features, train_targets, test_features
    else:
        return train_features, train_targets


def get_model(feature_cards, args):
    if args.model == 'FM':
        model = FactorizationMachine(feature_cards, factor_dim=args.factor_dim)
    if args.model == 'FFM':
        model = FieldAwareFactorizationMachine(feature_cards, factor_dim=args.factor_dim)
    if args.model == 'FNN':
        model = FMNeuralNetwork(feature_cards, args.factor_dim, [int(i) for i in args.fnn_hidden.split(',')])
    if args.model == 'AFM':
        model = AttentionalFactorizationMachine(feature_cards, args.factor_dim, args.afm_attn_size)
    return model


def main():
    feature_encoders = load_feature_encoders()
    feature_cards = [len(feature_encoders[i]) + 1 for i in range(1, 40)]

    train_features, train_targets, test_features = get_encoded_dataset(feature_encoders, get_testset=True)
    train_targets = train_targets.reshape(-1, 1)
    x_train, x_valid, y_train, y_valid = train_test_split(train_features, train_targets, test_size=.05,
                                                          random_state=42)
    del train_features, train_targets
    gc.collect()

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).\
            shuffle(buffer_size=args.steps_per_epoch).batch(args.batch_size)
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(args.batch_size)
    del x_train, y_train, x_valid, y_valid
    gc.collect()

    best_score = 0
    best_epoch = 0
    lr = args.learning_rate
    patience = args.patience
    earlystop = args.earlystop

    model = get_model(feature_cards, args)
    criterion = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            probas = tf.nn.sigmoid(logits)
            loss = criterion(y, probas)
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return probas, y, loss

    @tf.function
    def valid_step(x, y):
        probas = tf.nn.sigmoid(model(x, training=False))
        loss = criterion(y, probas)
        return probas, y, loss

    for epoch in range(1, args.max_epoch + 1):
        train_loss = tf.keras.metrics.Mean()
        train_auc = tf.keras.metrics.AUC()

        for step, (x, y) in enumerate(train_dataset.take(args.steps_per_epoch), 1):
            probas, y, loss = train_step(x, y)
            train_loss(loss)
            train_auc(y, probas)
            print('\repoch {:>4} step {:>8} train loss {:.4f} - auc {:.4f}'.format(
                epoch, step, train_loss.result(), train_auc.result()), end='', flush=True)
        print('\r\n', end='', flush=False)

        valid_loss = tf.keras.metrics.Mean()
        valid_auc = tf.keras.metrics.AUC()
        for step, (x, y) in enumerate(valid_dataset, 1):
            probas, y, loss = valid_step(x, y)
            valid_loss(loss)
            valid_auc(y, probas)
        valid_message = 'epoch {:>4} validation loss {:.4f} - auc {:.4f}'.format(
                epoch, valid_loss.result(), valid_auc.result())
        print(valid_message)
        with open(LOG_FILE, 'a+') as f:
            f.write(args.model + ' ' + valid_message + '\n')

        model.save_weights(os.path.join(MODEL_PATH, 'epoch_{}'.format(epoch)), save_format='tf')

        score = valid_auc.result()
        if score > best_score:
            best_score = score
            best_epoch = epoch
        else:
            if epoch - best_epoch > patience:
                lr *= .1
                optimizer.lr.assign(lr)
                print('--learning rate reduced to {}'.format(lr))
            if epoch - best_epoch > earlystop:
                print('--early stopping triggered')
                break


if __name__ == '__main__':
    main()

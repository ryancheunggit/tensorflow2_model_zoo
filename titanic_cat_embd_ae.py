"""Example program training categorical feature embedding with auto encoder network."""
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow import keras

TITANIC_CSV = 'data/titanic.csv'
CAT_COLUMNS = ['Pclass', 'Title']
NUM_COLUMNS = ['Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
EMBEDDING_DIMS = [1, 2]
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_AE_EPOCHS = 300
NUM_REG_EPOCHS = 300
VERBOSE = 1

np.random.seed(1412)
tf.random.set_seed(1412)


def _get_data():
    df = pd.read_csv(TITANIC_CSV)
    df['Title'] = df['Name'].map(lambda x: x.split('.')[0].lower())
    Title_counts = df['Title'].value_counts()
    to_bin = set(Title_counts.index[Title_counts < 10])
    df['Title'] = df['Title'].map(lambda x: 'other' if x in to_bin else x)
    df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'male' else 0)
    return df


def _get_embeddings(df, feature_names, embedding_dims):
    assert len(feature_names) == len(embedding_dims)
    embedding_columns = []
    one_hot_columns = []
    for name, embedding_dim in zip(feature_names, embedding_dims):
        feature_levels = feature_column.categorical_column_with_vocabulary_list(name, df[name].unique())
        embedding_columns.append(feature_column.embedding_column(feature_levels, embedding_dim))
        one_hot_columns.append(feature_column.indicator_column(feature_levels))
    return embedding_columns, one_hot_columns


class AutoEncoder(keras.Model):
    """Auto Encoder Network, used to train categorical feature embedding."""
    def __init__(self, feature_embedding, latent_dim, out_dim):
        super(AutoEncoder, self).__init__()
        self.lookup = feature_embedding
        self.encoder = keras.layers.Dense(latent_dim, activation='relu')
        self.decoder = keras.layers.Dense(units=out_dim, activation=None)

    def lookup(self, x):
        return self.lookup(x)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return tf.nn.sigmoid(self.decoder(z))

    def call(self, x):
        return self.decode(self.encode(self.lookup(x)))


class Classifier(keras.Model):
    """Classifier."""
    def __init__(self, feature_embedding, numeric_feature, embedding_trainable=False):
        super(Classifier, self).__init__()
        self.feature_embedding = feature_embedding
        self.feature_embedding.trainable = embedding_trainable
        self.numeric_feature = numeric_feature
        self.classifier = keras.Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.Dense(units=1, activation='sigmoid')
        ])

    def call(self, x, training=True):
        feature = tf.concat([self.numeric_feature(x), self.feature_embedding(x)], axis=-1)
        return self.classifier(feature, training=training)


# prepare data
df = _get_data()
embedding_columns, one_hot_columns = _get_embeddings(df, CAT_COLUMNS, EMBEDDING_DIMS)
feature_embedding = keras.layers.DenseFeatures(embedding_columns)
numeric_features = keras.layers.DenseFeatures([feature_column.numeric_column(name) for name in NUM_COLUMNS])


def _train_categorical_feature_embedding(verbose=1):
    cat_feature_dataset = tf.data.Dataset.from_tensor_slices(dict(df[CAT_COLUMNS])).\
        shuffle(buffer_size=len(df)).\
        batch(BATCH_SIZE)
    one_hot_encoder = keras.layers.DenseFeatures(one_hot_columns)
    cadinality = one_hot_encoder(next(iter(cat_feature_dataset.take(1)))).shape[1]
    ae_model = AutoEncoder(feature_embedding, latent_dim=2, out_dim=cadinality)
    criterion = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_metric = keras.metrics.Mean()
    accuracy_metric = keras.metrics.CategoricalAccuracy()

    # @tf.function  # not working issue: https://github.com/tensorflow/tensorflow/issues/27086
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            recovered = ae_model(x_batch)
            loss = criterion(y_batch, recovered)
        grad = tape.gradient(loss, ae_model.trainable_variables)
        optimizer.apply_gradients(zip(grad, ae_model.trainable_variables))
        loss_metric(loss)
        accuracy_metric(y_batch, recovered)

    message_template = 'epoch {:>3} time {} sec / epoch cce {:.4f} reconstruct error {:4.2f}%'
    t0 = datetime.now()
    for epoch in range(NUM_AE_EPOCHS):
        for idx, x_batch in enumerate(cat_feature_dataset):
            y_batch = one_hot_encoder(x_batch)
            train_step(x_batch, y_batch)
        if (epoch % 10 == 0 or epoch == NUM_AE_EPOCHS - 1) and verbose:
            t1 = datetime.now()
            print(message_template.format(
                epoch + 1, (t1 - t0).seconds,
                loss_metric.result(), 100 * (1 - accuracy_metric.result())
            ))
            t0 = datetime.now()


_train_categorical_feature_embedding(verbose=VERBOSE)

if VERBOSE:
    print('--- take a look at learned categorical feature embeddings.')
    print('feature_levels: ')
    print(one_hot_columns)
    print('trained categorical embedding')
    print(feature_embedding.variables)


def _train_classifier(verbose=1):
    train, test = train_test_split(df, test_size=.2, random_state=42)
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train), train['Survived'])).\
        shuffle(buffer_size=len(train)).\
        batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test), test['Survived'])).\
        shuffle(buffer_size=len(test)).\
        batch(BATCH_SIZE)
    classifier = Classifier(feature_embedding, numeric_features)

    criterion = keras.losses.CategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    test_accuracy = keras.metrics.CategoricalAccuracy()

    # @tf.function scope name error if enabled, why?
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            out = classifier(x_batch, training=True)
            loss = criterion(y_batch, out)
        grad = tape.gradient(loss, classifier.trainable_variables)
        optimizer.apply_gradients(zip(grad, classifier.trainable_variables))

    # @tf.function
    def valid_step(x_batch, y_batch):
        out = classifier(x_batch, training=False)
        test_accuracy(y_batch, out > .5)  # why I need this hard convertion?

    for epoch in range(NUM_REG_EPOCHS):
        t0 = datetime.now()
        for idx, (x_batch, y_batch) in enumerate(train_dataset):
            train_step(x_batch, y_batch)

        for idx, (x_batch, y_batch) in enumerate(test_dataset):
            valid_step(x_batch, y_batch)
        message_template = 'epoch {:>3} time {} sec / epoch test acc {:4.2f}%'
        t1 = datetime.now()
        if (epoch % 10 == 0 or epoch == NUM_REG_EPOCHS - 1) and verbose:
            print(message_template.format(
                epoch + 1, (t1 - t0).seconds,
                test_accuracy.result() * 100
            ))


_train_classifier(verbose=VERBOSE)

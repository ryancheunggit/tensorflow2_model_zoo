"""EfficientNet. Based on https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet."""
import math
import numpy as np
import tensorflow as tf
from collections import namedtuple
from tensorflow import nn
from tensorflow import keras
from tensorflow.keras import layers
from activation import swish

GlobalParams = namedtuple('GlobalParams', [
    'bn_momentum', 'bn_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate'
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)


BlockArgs = namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'out_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random.normal(shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)


def round_filters(filters, global_params):
    original_filters = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < .9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, is_training, drop_connect_rate):
    """Drop connection."""
    if not is_training:
        return inputs
    keep_prob = 1.0 - drop_connect_rate
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform((batch_size, 1, 1, 1), dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.floor(tf.divide(inputs, keep_prob)) * binary_tensor
    return output


class MBConvBlock(object):
    """Mobile Inverted Residual Bottleneck block."""
    def __init__(self, block_args, global_params):
        self._block_args = block_args
        self._bn_momentum = global_params.bn_momentum
        self._bn_epsilon = global_params.bn_epsilon
        self._data_format = global_params.data_format
        self._channel_axis = 1 if self._data_format == 'channels_first' else - 1
        self._spatial_dims = (2, 3) if self._data_format == 'channels_first' else (1, 2)
        self.has_se = (0 < getattr(self._block_args, 'se_ratio', 0) <= 1)
        self.endpoints = None
        self._build()

    def block_args(self):
        return self._block_args

    def _build(self):
        filters = self._block_args.input_filters * self._block_args.expand_ratio
        kernel_size = self._block_args.kernel_size

        # Expand
        if self._block_args.expand_ratio != 1:
            self._expand_conv = layers.Conv2D(
                filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,
                data_format=self._data_format, kernel_initializer=conv_kernel_initializer
            )
            self._bn0 = layers.BatchNormalization(
                axis=self._channel_axis, momentum=self._bn_momentum, epsilon=self._bn_epsilon
            )

        # Depthwise Conv
        self._depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=(kernel_size, kernel_size), strides=self._block_args.strides, padding='same', use_bias=False,
            data_format=self._data_format, depthwise_initializer=conv_kernel_initializer
        )
        self._bn1 = layers.BatchNormalization(
            axis=self._channel_axis, momentum=self._bn_momentum, epsilon=self._bn_epsilon
        )

        # Squeeze-Excite
        if self.has_se:
            num_reduced_filters = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se_reduce = layers.Conv2D(
                filters=num_reduced_filters, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True,
                data_format=self._data_format, kernel_initializer=conv_kernel_initializer
            )
            self._se_expand = layers.Conv2D(
                filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True,
                data_format=self._data_format, kernel_initializer=conv_kernel_initializer
            )

        # Output
        filters = self._block_args.out_filters
        self._project_conv = layers.Conv2D(
            filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False,
            data_format=self._data_format, kernel_initializer=conv_kernel_initializer
        )
        self._bn2 = layers.BatchNormalization(
            axis=self._channel_axis, momentum=self._bn_momentum, epsilon=self._bn_epsilon
        )

    def _call_se(self, input_tensor):
        se_tensor = tf.reduce_mean(input_tensor, self._spatial_dims, keepdims=True)
        se_tensor = self._se_expand(swish(self._se_reduce(se_tensor)))
        return tf.sigmoid(se_tensor) * input_tensor

    def call(self, inputs, training=True, drop_connect_rate=None):
        # Expand
        if self._block_args.expand_ratio != 1:
            x = swish(self._bn0(self._expand_conv(inputs), training=training))
        else:
            x = inputs

        # Depthwise Conv
        x = swish(self._bn1(self._depthwise_conv(x), training=training))

        if self.has_se:
            x = self._call_se(x)

        self.endpoints = {'expansion_output': x}

        x = self._bn2(self._project_conv(x), training=training)

        if self._block_args.id_skip:
            if all(s == 1 for s in self._block_args.strides) and (
                self._block_args.input_filters == self._block_args.out_filters
            ):
                if drop_connect_rate:
                    x = drop_connect(x, training, drop_connect_rate)
                x = tf.add(x, inputs)
        return x


class EfficientNet(keras.Model):
    """EfficientNet, based on https://arxiv.org/abs/1807.11626."""
    def __init__(self, blocks_args=None, global_params=None):
        super(EfficientNet, self).__init__()
        self._blocks_args = blocks_args
        self._global_params = global_params
        self.endpoints = None
        self._build()

    def _build(self):
        # Blocks
        self._blocks = []
        for block_args in self._blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                out_filters=round_filters(block_args.out_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.out_filters, strides=(1, 1))

            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))
        bn_momentum = self._global_params.bn_momentum
        bn_epsilon = self._global_params.bn_epsilon
        if self._global_params.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        # Stem
        self._conv_stem = layers.Conv2D(
            filters=round_filters(32, self._global_params), kernel_size=(3, 3), strides=(2, 2), padding='same',
            use_bias=False, data_format=self._global_params.data_format,
            kernel_initializer=conv_kernel_initializer
        )
        self._bn0 = layers.BatchNormalization(
            axis=channel_axis, momentum=bn_momentum, epsilon=bn_epsilon
        )

        # Head
        self._conv_head = layers.Conv2D(
            filters=round_filters(1280, self._global_params), kernel_size=(1, 1), strides=(3, 3), padding='same',
            use_bias=False, data_format=self._global_params.data_format,
            kernel_initializer=conv_kernel_initializer
        )
        self._bn1 = layers.BatchNormalization(
            axis=channel_axis, momentum=bn_momentum, epsilon=bn_epsilon
        )

        self._avg_pooling = layers.GlobalAveragePooling2D(data_format=self._global_params.data_format)
        self._fc = layers.Dense(self._global_params.num_classes, kernel_initializer=dense_kernel_initializer)

        if self._global_params.dropout_rate > 0:
            self._dropout = layers.Dropout(self._global_params.dropout_rate)
        else:
            self._dropout = None

    def call(self, inputs, training=True, features_only=None):
        outputs = None
        self.endpoints = {}

        # Stem
        outputs = swish(self._bn0(self._conv_stem(inputs), training=training))
        self.endpoints['stem'] = outputs

        # Blocks
        reduction_idx = 0
        for idx, block in enumerate(self._blocks):
            is_reduction = False
            if ((idx == len(self._blocks) - 1) or (self._blocks[idx + 1].block_args().strides[0] > 1)):
                is_reduction = True
                reduction_idx += 1

            drop_rate = self._global_params.drop_connect_rate
            if drop_rate:
                drop_rate *= float(idx) / len(self._blocks)
            outputs = block.call(outputs, training=training, drop_connect_rate=drop_rate)
            self.endpoints['block_{}'.format(idx)] = outputs

            if is_reduction:
                self.endpoints['reduction_{}'.format(reduction_idx)] = outputs

            if block.endpoints:
                for k, v in block.endpoints.items():
                    self.endpoints['block_{}/{}'.format(idx, k)] = v
                    if is_reduction:
                        self.endpoints['reduction_{}/{}'.format(reduction_idx, k)] = v
        self.endpoints['global_pool'] = outputs

        # Head
        if not features_only:
            outputs = swish(self._bn1(self._conv_head(outputs), training=training))
            outputs = self._avg_pooling(outputs)
            if self._dropout:
                outputs = self._dropout(outputs, training=training)
            self.endpoints['global_pool'] = outputs
            outputs = self._fc(outputs)
            self.endpoints['head'] = outputs
        return outputs


def get_efficientnet_base_args(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2, drop_connect_rate=0.2):
    blocks_args = [
        BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, out_filters=16, expand_ratio=1, id_skip=True, se_ratio=.25, strides=(1, 1)),
        BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, out_filters=24, expand_ratio=6, id_skip=True, se_ratio=.25, strides=(2, 2)),
        BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, out_filters=40, expand_ratio=6, id_skip=True, se_ratio=.25, strides=(2, 2)),
        BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, out_filters=80, expand_ratio=6, id_skip=True, se_ratio=.25, strides=(2, 2)),
        BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, out_filters=112, expand_ratio=6, id_skip=True, se_ratio=.25, strides=(1, 1)),
        BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, out_filters=192, expand_ratio=6, id_skip=True, se_ratio=.25, strides=(2, 2)),
        BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, out_filters=320, expand_ratio=6, id_skip=True, se_ratio=.25, strides=(1, 1))
    ]
    global_params = GlobalParams(
        bn_momentum=0.99,
        bn_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        data_format='channels_last',
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None
    )
    return blocks_args, global_params


def build_base_efficientnet():
    blocks_args, global_params = get_efficientnet_base_args(width_coefficient=1, depth_coefficient=1, dropout_rate=.2)
    model = EfficientNet(blocks_args, global_params)
    return model


model = build_base_efficientnet()
x = tf.random.uniform((32, 224, 224, 3))
out = model(x)
print(out)

# Taken from https://raw.githubusercontent.com/google-research/big_transfer/master/bit_tf2/
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ResNet architecture as used in BiT."""

import tensorflow as tf
from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from . import normalization
from ....common import StandardizedConv2D


@register_keras_serializable(package='SegMe>Backbone>BigTransfer')
class PaddingFromKernelSize(layers.Layer):
    """Layer that adds padding to an image taking into a given kernel size."""

    def __init__(self, kernel_size, dilation_rate=1, **kwargs):
        super(PaddingFromKernelSize, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        pad_total = (kernel_size - 1) * dilation_rate
        self._pad_beg = pad_total // 2
        self._pad_end = pad_total - self._pad_beg

    def compute_output_shape(self, input_shape):
        batch_size, height, width, channels = tf.TensorShape(input_shape).as_list()
        if height is not None:
            height = height + self._pad_beg + self._pad_end
        if width is not None:
            width = width + self._pad_beg + self._pad_end
        return tf.TensorShape((batch_size, height, width, channels))

    def call(self, x):
        padding = [
            [0, 0],
            [self._pad_beg, self._pad_end],
            [self._pad_beg, self._pad_end],
            [0, 0]]
        return tf.pad(x, padding)

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate
        })

        return config


@register_keras_serializable(package='SegMe>Backbone>BigTransfer')
class BottleneckV2Unit(layers.Layer):
    """Implements a standard ResNet's unit (version 2).
    """

    def __init__(self, num_filters, stride=1, dilation=1, **kwargs):
        """Initializer.

        Args:
          num_filters: number of filters in the bottleneck.
          stride: specifies block's stride.
          **kwargs: other layers.Layer keyword arguments.
        """
        super(BottleneckV2Unit, self).__init__(**kwargs)
        self._num_filters = num_filters
        self._stride = stride
        self._dilation = dilation

    def build(self, input_shape):
        self._unit_a = models.Sequential([
            normalization.GroupNormalization(name="group_norm"),
            layers.ReLU(),
        ], name="a")
        self._unit_a_conv = StandardizedConv2D(
            filters=self._num_filters,
            kernel_size=1,
            use_bias=False,
            padding="VALID",
            trainable=self.trainable,
            name="a/standardized_conv2d")

        self._unit_b = models.Sequential([
            normalization.GroupNormalization(name="group_norm"),
            layers.ReLU(),
            PaddingFromKernelSize(kernel_size=3, dilation_rate=self._dilation),
            StandardizedConv2D(
                filters=self._num_filters,
                kernel_size=3,
                strides=self._stride,
                dilation_rate=self._dilation,
                use_bias=False,
                padding="VALID",
                trainable=self.trainable,
                name="standardized_conv2d")
        ], name="b")

        self._unit_c = models.Sequential([
            normalization.GroupNormalization(name="group_norm"),
            layers.ReLU(),
            StandardizedConv2D(
                filters=4 * self._num_filters,
                kernel_size=1,
                use_bias=False,
                padding="VALID",
                trainable=self.trainable,
                name="standardized_conv2d")
        ], name="c")

        # Add projection layer if necessary.
        self._proj = None
        input_shape = tf.TensorShape(input_shape).as_list()
        if (self._stride > 1) or (4 * self._num_filters != input_shape[-1]):
            self._proj = StandardizedConv2D(
                filters=4 * self._num_filters,
                kernel_size=1,
                strides=self._stride,
                use_bias=False,
                padding="VALID",
                trainable=self.trainable,
                name="a/proj/standardized_conv2d")
        self.built = True

    def compute_output_shape(self, input_shape):
        current_shape = self._unit_a.compute_output_shape(input_shape)
        current_shape = self._unit_a_conv.compute_output_shape(current_shape)
        current_shape = self._unit_b.compute_output_shape(current_shape)
        current_shape = self._unit_c.compute_output_shape(current_shape)
        return current_shape

    def call(self, x):
        x_shortcut = x
        # Unit "a".
        x = self._unit_a(x)
        if self._proj is not None:
            x_shortcut = self._proj(x)
        x = self._unit_a_conv(x)
        # Unit "b".
        x = self._unit_b(x)
        # Unit "c".
        x = self._unit_c(x)

        return x + x_shortcut

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters': self._num_filters,
            'stride': self._stride,
            'dilation': self._dilation
        })

        return config


def ResnetV2(x, num_units=(3, 4, 6, 3), num_outputs=1000, filters_factor=4, strides=(1, 2, 2, 2)):
    num_blocks = len(num_units)
    num_filters = tuple(16 * filters_factor * 2 ** b for b in range(num_blocks))

    _root = [
        PaddingFromKernelSize(7),
        StandardizedConv2D(
            filters=num_filters[0],
            kernel_size=7,
            strides=2,
            use_bias=False,
            name="standardized_conv2d"),
        PaddingFromKernelSize(3),
        layers.MaxPool2D(
            pool_size=3,
            strides=2,
            padding="valid")
    ]

    _blocks = []
    for b, (f, u, s) in enumerate(zip(num_filters, num_units, strides), 1):
        _blocks.append(models.Sequential([
            BottleneckV2Unit(
                num_filters=f,
                stride=(s if i == 1 else 1),
                name="unit%02d" % i) for i in range(1, u + 1)
        ], name="block{}".format(b)))
        _blocks.append(layers.Activation('linear', name="block{}_out".format(b)))

    _pre_head = [
        normalization.GroupNormalization(name="group_norm"),
        layers.ReLU(name='head_relu'),
        layers.GlobalAveragePooling2D()
    ]

    _head = None
    if num_outputs:
        _head = layers.Dense(
            units=num_outputs,
            use_bias=True,
            kernel_initializer="zeros",
            name="head/dense")

    # x = _root(x)
    for block in _root:
        x = block(x)
    for block in _blocks:
        x = block(x)
    for layer in _pre_head:
        x = layer(x)
    if _head is not None:
        x = _head(x)

    return x


KNOWN_MODELS = {
    f'{bit}-R{l}x{w}': f'gs://bit_models/{bit}-R{l}x{w}.h5'
    for bit in ['BiT-S', 'BiT-M']
    for l, w in [(50, 1), (50, 3), (101, 1), (101, 3), (152, 4)]
}

NUM_UNITS = {
    k: (3, 4, 6, 3) if 'R50' in k else
    (3, 4, 23, 3) if 'R101' in k else
    (3, 8, 36, 3)
    for k in KNOWN_MODELS
}

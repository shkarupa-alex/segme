# Taken from https://github.com/tensorflow/addons/blob/master/tensorflow_addons/layers/adaptive_pooling.py
#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from keras.utils.conv_utils import normalize_tuple


class AdaptivePooling(layers.Layer):
    # TODO: wait for https://github.com/tensorflow/addons/pull/2322

    def __init__(self, reduce_function, output_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.reduce_function = reduce_function
        self.output_size = normalize_tuple(output_size, 2, 'output_size')

    def call(self, inputs, *args):
        inputs_shape = tf.shape(inputs)
        inputs_shape1, inputs_shape2 = inputs_shape[1], inputs_shape[2]
        start_points_x = tf.cast(
            (tf.range(self.output_size[0], dtype=tf.float32)
             * tf.cast(inputs_shape1 / self.output_size[0], tf.float32)),
            tf.int32)
        end_points_x = tf.cast(
            tf.math.ceil(
                (tf.range(self.output_size[0], dtype=tf.float32) + 1)
                * tf.cast(inputs_shape1 / self.output_size[0], tf.float32)),
            tf.int32)
        start_points_y = tf.cast(
            (tf.range(self.output_size[1], dtype=tf.float32)
             * tf.cast(inputs_shape2 / self.output_size[1], tf.float32)),
            tf.int32)
        end_points_y = tf.cast(
            tf.math.ceil((tf.range(self.output_size[1], dtype=tf.float32) + 1)
                         * tf.cast(inputs_shape2 / self.output_size[1], tf.float32)),
            tf.int32)

        pooled = []
        for idx in range(self.output_size[0]):
            pooled.append(self.reduce_function(
                inputs[:, start_points_x[idx]: end_points_x[idx], :, :],
                axis=1,
                keepdims=True,
            ))
        x_pooled = tf.concat(pooled, axis=1)

        pooled = []
        for idx in range(self.output_size[1]):
            pooled.append(self.reduce_function(
                x_pooled[:, :, start_points_y[idx]: end_points_y[idx], :],
                axis=2,
                keepdims=True,
            ))
        y_pooled = tf.concat(pooled, axis=2)

        return y_pooled

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_size[0], self.output_size[1], input_shape[3]

    def get_config(self):
        config = super().get_config()
        config.update({'output_size': self.output_size})

        return config


@register_keras_serializable(package='SegMe')
class AdaptiveAveragePooling(AdaptivePooling):
    def __init__(self, output_size, **kwargs):
        super().__init__(reduce_function=tf.reduce_mean, output_size=output_size, **kwargs)


@register_keras_serializable(package='SegMe')
class AdaptiveMaxPooling(AdaptivePooling):
    def __init__(self, output_size, **kwargs):
        super().__init__(reduce_function=tf.reduce_max, output_size=output_size, **kwargs)

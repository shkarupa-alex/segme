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
from keras.utils.control_flow_util import smart_cond
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from keras.utils.conv_utils import normalize_tuple


@register_keras_serializable(package='SegMe>Common')
class AdaptivePooling(layers.Layer):
    # TODO: wait for https://github.com/tensorflow/addons/pull/2322

    def __init__(self, reduce_function, output_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.reduce_function = reduce_function
        self.output_size = normalize_tuple(output_size, 2, 'output_size')

    def case_global(self, inputs):
        return self.reduce_function(inputs, axis=[1, 2], keepdims=True)

    def case_divisible(self, inputs):
        split_x = tf.split(inputs, self.output_size[0], axis=1)
        split_x = tf.stack(split_x, axis=1)
        split_y = tf.split(split_x, self.output_size[1], axis=3)
        split_y = tf.stack(split_y, axis=3)

        outputs = self.reduce_function(split_y, axis=[2, 4])

        return outputs

    def case_nondivisible(self, inputs):
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

    def call(self, inputs, *args, **kwargs):
        if (1, 1) == self.output_size:
            return self.case_global(inputs)

        height, width = tf.unstack(tf.shape(inputs)[1:3])
        pad_h = (self.output_size[0] - height % self.output_size[0]) % self.output_size[0]
        pad_w = (self.output_size[1] - width % self.output_size[1]) % self.output_size[1]

        # Hack to allow build non-divisible branch
        inputs_ = tf.pad(inputs, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

        outputs = smart_cond(
            (pad_h == 0) & (pad_w == 0),
            lambda: self.case_divisible(inputs_),
            lambda: self.case_nondivisible(inputs))

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_size[0], self.output_size[1], input_shape[3]

    def get_config(self):
        config = super().get_config()
        config.update({'output_size': self.output_size})

        return config


@register_keras_serializable(package='SegMe>Common')
class AdaptiveAveragePooling(AdaptivePooling):
    def __init__(self, output_size, **kwargs):
        super().__init__(reduce_function=tf.reduce_mean, output_size=output_size, **kwargs)


@register_keras_serializable(package='SegMe>Common')
class AdaptiveMaxPooling(AdaptivePooling):
    def __init__(self, output_size, **kwargs):
        super().__init__(reduce_function=tf.reduce_max, output_size=output_size, **kwargs)

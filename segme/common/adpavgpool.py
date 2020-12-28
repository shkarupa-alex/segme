import tensorflow as tf
from tensorflow.keras import layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from tensorflow.python.keras.utils.conv_utils import normalize_tuple


@utils.register_keras_serializable(package='SegMe')
class AdaptiveAveragePooling(layers.Layer):
    def __init__(self, output_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)
        self.output_size = normalize_tuple(output_size, 2, 'output_size')

    def call(self, inputs, *args):
        start_points_x = tf.cast(
            (
                    tf.range(self.output_size[0], dtype=tf.float32)
                    * tf.cast((tf.shape(inputs)[1] / self.output_size[0]), tf.float32)
            ),
            tf.int32,
        )
        end_points_x = tf.cast(
            tf.math.ceil(
                (
                        (tf.range(self.output_size[0], dtype=tf.float32) + 1)
                        * tf.cast(
                    (tf.shape(inputs)[1] / self.output_size[0]), tf.float32
                )
                )
            ),
            tf.int32,
        )

        start_points_y = tf.cast(
            (
                    tf.range(self.output_size[1], dtype=tf.float32)
                    * tf.cast((tf.shape(inputs)[2] / self.output_size[1]), tf.float32)
            ),
            tf.int32,
        )
        end_points_y = tf.cast(
            tf.math.ceil(
                (
                        (tf.range(self.output_size[1], dtype=tf.float32) + 1)
                        * tf.cast(
                    (tf.shape(inputs)[2] / self.output_size[1]), tf.float32
                )
                )
            ),
            tf.int32,
        )
        pooled = []
        for idx in range(self.output_size[0]):
            pooled.append(
                tf.reduce_mean(
                    inputs[:, start_points_x[idx]: end_points_x[idx], :, :],
                    axis=1,
                    keepdims=True,
                )
            )
        x_pooled = tf.concat(pooled, axis=1)

        pooled = []
        for idx in range(self.output_size[1]):
            pooled.append(
                tf.reduce_mean(
                    x_pooled[:, :, start_points_y[idx]: end_points_y[idx], :],
                    axis=2,
                    keepdims=True,
                )
            )
        y_pooled = tf.concat(pooled, axis=2)

        return y_pooled

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_size[0], self.output_size[1], input_shape[3]

    def get_config(self):
        config = super().get_config()
        config.update({'output_size': self.output_size})

        return config

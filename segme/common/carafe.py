import tensorflow as tf
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.resize import NearestInterpolation
from segme.common.shape import get_shape


@register_keras_serializable(package="SegMe>Common>Align>FADE")
class CarafeConvolution(layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            InputSpec(ndim=4),  # features
            InputSpec(ndim=4),
        ]  # mask

        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.channels = [shape[-1] for shape in input_shape]
        if None in self.channels:
            raise ValueError(
                "Channel dimension of the inputs should be defined. Found `None`."
            )
        self.input_spec = [
            InputSpec(ndim=4, axes={-1: self.channels[0]}),
            InputSpec(ndim=4, axes={-1: self.channels[1]}),
        ]

        self.internear = NearestInterpolation(dtype=self.dtype_policy)

        self.group_size = self.channels[1] // (self.kernel_size**2)
        if (
            self.group_size < 1
            or self.channels[1] != self.group_size * self.kernel_size**2
        ):
            raise ValueError("Wrong mask channel dimension.")

        if self.channels[0] % self.group_size:
            raise ValueError("Unable to split features into groups.")

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        features, masks = inputs

        (batch, height, width), _ = get_shape(masks, axis=[0, 1, 2])
        output_shape = self.compute_output_shape([features.shape, masks.shape])

        features = tf.image.extract_patches(
            features,
            [1, self.kernel_size, self.kernel_size, 1],
            [1] * 4,
            [1] * 4,
            "SAME",
        )

        if False and 1 == self.group_size:
            features = self.internear([features, masks])
            features = tf.reshape(
                features,
                (batch, height, width, self.kernel_size**2, self.channels[0]),
            )

            masks = tf.nn.softmax(masks)[..., None]

            outputs = tf.matmul(features, masks, transpose_a=True)
        else:
            features_shape0, _ = get_shape(features)
            features_shape1 = features_shape0[:-1] + [
                self.kernel_size**2,
                self.channels[0],
            ]
            features = tf.reshape(features, features_shape1)
            features = tf.transpose(features, [0, 1, 2, 4, 3])
            features = tf.reshape(features, features_shape0)
            features = self.internear([features, masks])
            features = tf.reshape(
                features,
                (
                    batch,
                    height,
                    width,
                    self.group_size,
                    self.channels[0] // self.group_size,
                    self.kernel_size**2,
                ),
            )

            masks = tf.reshape(
                masks,
                (batch, height, width, self.group_size, self.kernel_size**2),
            )
            masks = tf.nn.softmax(masks)[..., None]

            outputs = tf.matmul(features, masks)

        outputs = tf.reshape(outputs, (batch, height, width, self.channels[0]))
        outputs.set_shape(output_shape)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (self.channels[0],)

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})

        return config

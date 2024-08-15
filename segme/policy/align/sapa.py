import tensorflow as tf
from keras.src import layers
from keras.src.saving import register_keras_serializable
from keras.src.layers.input_spec import InputSpec
from segme.common.convnormact import Conv
from segme.common.convnormact import Norm
from segme.common.resize import NearestInterpolation
from segme.common.shape import get_shape


@register_keras_serializable(package="SegMe>Policy>Align>SAPA")
class SapaFeatureAlignment(layers.Layer):
    """
    Proposed in "SAPA: Similarity-Aware Point Affiliation for Feature Upsampling"
    https://arxiv.org/pdf/2209.12866.pdf
    """

    def __init__(self, filters, kernel_size=5, embedding_size=64, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            InputSpec(ndim=4),  # fine
            InputSpec(ndim=4),
        ]  # coarse

        self.filters = filters
        self.kernel_size = kernel_size
        self.embedding_size = embedding_size

    def build(self, input_shape):
        self.norm_fine = Norm(policy="conv-ln1em5-relu", dtype=self.dtype_policy)
        self.norm_fine.build(input_shape[0])

        self.norm_coarse = Norm(policy="conv-ln1em5-relu", dtype=self.dtype_policy)
        self.norm_coarse.build(input_shape[1])

        self.intnear = NearestInterpolation(None, dtype=self.dtype_policy)

        self.query_gate = layers.Conv2D(1, 1, activation="sigmoid", dtype=self.dtype_policy)
        self.query_gate.build(input_shape[1])

        self.query_fine = Conv(self.embedding_size, 1, dtype=self.dtype_policy)
        self.query_fine.build(input_shape[0])

        self.query_coarse = Conv(self.embedding_size, 1, dtype=self.dtype_policy)
        self.query_coarse.build(input_shape[1])

        self.key = Conv(self.embedding_size, 1, dtype=self.dtype_policy)
        self.key.build(input_shape[1])

        self.value = Conv(self.filters, 1, dtype=self.dtype_policy)
        self.value.build(input_shape[1])

        self.attend = LocalAttention(self.kernel_size, dtype=self.dtype_policy)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        fine, coarse = inputs

        fine = self.norm_fine(fine)
        coarse = self.norm_coarse(coarse)

        gate = self.query_gate(coarse)
        gate = self.intnear([gate, fine])

        query = self.query_fine(fine) * gate + self.intnear(
            [self.query_coarse(coarse), fine]
        ) * (1.0 - gate)
        key = self.key(coarse)
        value = self.value(coarse)

        outputs = self.attend([query, key, value])

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "embedding_size": self.embedding_size,
            }
        )

        return config


@register_keras_serializable(package="SegMe>Policy>Align>SAPA")
class LocalAttention(layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            InputSpec(ndim=4),  # query
            InputSpec(ndim=4),  # key
            InputSpec(ndim=4),
        ]  # value

        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.channels = [shape[-1] for shape in input_shape]
        if None in self.channels:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )
        if self.channels[0] != self.channels[1]:
            raise ValueError(
                "Channel dimension of the query and key should be equal."
            )
        self.input_spec = [
            InputSpec(ndim=4, axes={-1: self.channels[0]}),
            InputSpec(ndim=4, axes={-1: self.channels[1]}),
            InputSpec(ndim=4, axes={-1: self.channels[2]}),
        ]

        self.patch_kwargs = {
            "sizes": [1, self.kernel_size, self.kernel_size, 1],
            "strides": [1] * 4,
            "rates": [1] * 4,
            "padding": "SAME",
        }

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        query, key, value = inputs
        shape = self.compute_output_shape([i.shape for i in inputs])

        (q_batch, q_height, q_width), _ = get_shape(query, axis=[0, 1, 2])
        (k_batch, k_height, k_width), _ = get_shape(key, axis=[0, 1, 2])
        (v_batch, v_height, v_width), _ = get_shape(value, axis=[0, 1, 2])

        assert_batch = tf.assert_equal(
            (q_batch == k_batch) & (k_batch == v_batch),
            True,
            "Batch dimension of the query, key and value should be equal.",
        )
        assert_size = tf.assert_equal(
            (k_height == v_height) & (k_width == v_width),
            True,
            "Spatial dimensions of the key and value should be equal.",
        )
        assert_scale = tf.assert_equal(
            (q_height % k_height == 0) & (q_width % k_width == 0),
            True,
            "Query-to-key/value scale value should be integer.",
        )
        with tf.control_dependencies([assert_batch, assert_size, assert_scale]):
            h_scale = q_height // k_height
            w_scale = q_width // k_width

        query = tf.reshape(
            query,
            [q_batch, k_height, h_scale, k_width, w_scale, self.channels[0]],
        )

        key = tf.image.extract_patches(key, **self.patch_kwargs)
        key = tf.reshape(
            key,
            [k_batch, k_height, k_width, self.kernel_size**2, self.channels[1]],
        )

        value = tf.image.extract_patches(value, **self.patch_kwargs)
        value = tf.reshape(
            value,
            [v_batch, v_height, v_width, self.kernel_size**2, self.channels[2]],
        )

        attention = tf.einsum("ijklmn,ijlon->ijklmo", query, key)
        attention = tf.nn.softmax(attention)

        outputs = tf.einsum("ijklmn,ijlno->ijklmo", attention, value)
        outputs = tf.reshape(
            outputs, [v_batch, q_height, q_width, self.channels[2]]
        )
        outputs.set_shape(shape)

        # query = tf.transpose(query, [0, 1, 3, 2, 4, 5])
        # query = tf.reshape(query, [q_batch, k_height, k_width, h_scale * w_scale, self.channels[0]])
        # attention = tf.matmul(query, key, transpose_b=True)
        # attention = tf.nn.softmax(attention)
        # outputs = tf.matmul(attention, value)
        # outputs = tf.reshape(outputs, [v_batch, v_height, v_width, h_scale, w_scale, self.channels[2]])
        # outputs = tf.transpose(outputs, [0, 1, 3, 2, 4, 5])

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + input_shape[2][-1:]

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})

        return config

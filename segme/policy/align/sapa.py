from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.convnormact import Conv
from segme.common.convnormact import Norm
from segme.common.resize import NearestInterpolation
from segme.ops import extract_patches


@register_keras_serializable(package="SegMe>Policy>Align>SAPA")
class SapaFeatureAlignment(layers.Layer):
    """
    Proposed in "SAPA: Similarity-Aware Point Affiliation for Feature
    Upsampling"
    https://arxiv.org/pdf/2209.12866
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
        self.norm_fine = Norm(
            policy="conv-ln1em5-relu", dtype=self.dtype_policy
        )
        self.norm_fine.build(input_shape[0])

        self.norm_coarse = Norm(
            policy="conv-ln1em5-relu", dtype=self.dtype_policy
        )
        self.norm_coarse.build(input_shape[1])

        self.intnear = NearestInterpolation(None, dtype=self.dtype_policy)

        self.query_gate = layers.Conv2D(
            1, 1, activation="sigmoid", dtype=self.dtype_policy
        )
        self.query_gate.build(input_shape[1])

        self.query_fine = Conv(self.embedding_size, 1, dtype=self.dtype_policy)
        self.query_fine.build(input_shape[0])

        self.query_coarse = Conv(
            self.embedding_size, 1, dtype=self.dtype_policy
        )
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
            "sizes": [self.kernel_size, self.kernel_size],
            "strides": [1, 1],
            "rates": [1, 1],
            "padding": "same",
        }

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        query, key, value = inputs
        shape = self.compute_output_shape([i.shape for i in inputs])

        q_batch, q_height, q_width = ops.shape(query)[:3]
        k_batch, k_height, k_width = ops.shape(key)[:3]
        v_batch, v_height, v_width = ops.shape(value)[:3]

        if (
            isinstance(q_batch, int)
            and isinstance(k_batch, int)
            and isinstance(v_batch, int)
            and not (q_batch == k_batch == v_batch)
        ):
            raise ValueError(
                "Batch dimension of the query, key and value should be equal."
            )
        if (
            isinstance(k_height, int)
            and isinstance(v_height, int)
            and isinstance(k_width, int)
            and isinstance(v_width, int)
            and (k_height != v_height or k_width != v_width)
        ):
            raise ValueError(
                "Spatial dimensions of the key and value should be equal."
            )
        if (
            isinstance(q_height, int)
            and isinstance(k_height, int)
            and isinstance(q_width, int)
            and isinstance(k_width, int)
            and (q_height % k_height != 0 or q_width % k_width != 0)
        ):
            raise ValueError(
                "Query-to-key/value scale value should be integer."
            )

        h_scale = q_height // k_height
        w_scale = q_width // k_width

        query = ops.reshape(
            query,
            [q_batch, k_height, h_scale, k_width, w_scale, self.channels[0]],
        )

        key = extract_patches(key, **self.patch_kwargs)
        key = ops.reshape(
            key,
            [k_batch, k_height, k_width, self.kernel_size**2, self.channels[1]],
        )

        value = extract_patches(value, **self.patch_kwargs)
        value = ops.reshape(
            value,
            [v_batch, v_height, v_width, self.kernel_size**2, self.channels[2]],
        )

        attention = ops.einsum("ijklmn,ijlon->ijklmo", query, key)
        attention = ops.softmax(attention)

        outputs = ops.einsum("ijklmn,ijlno->ijklmo", attention, value)
        outputs = ops.reshape(
            outputs, [v_batch, q_height, q_width, self.channels[2]]
        )
        outputs.set_shape(shape)

        # query = ops.transpose(query, [0, 1, 3, 2, 4, 5])
        # query = ops.reshape(query, [
        #   q_batch, k_height, k_width, h_scale * w_scale, self.channels[0]])
        # attention = ops.matmul(query, ops.moveaxis(key, -1, -2))
        # attention = ops.softmax(attention)
        # outputs = ops.matmul(attention, value)
        # outputs = ops.reshape(outputs, [
        #   v_batch, v_height, v_width, h_scale, w_scale, self.channels[2]])
        # outputs = ops.transpose(outputs, [0, 1, 3, 2, 4, 5])

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + input_shape[2][-1:]

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})

        return config

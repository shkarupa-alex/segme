from keras.src import layers
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable
from keras.src.utils.argument_validation import standardize_tuple

from segme.common.convnormact import Conv
from segme.common.convnormact import Norm
from segme.ops import adaptive_average_pooling_2d


@register_keras_serializable(package="SegMe>Common")
class AdaptiveAveragePooling(layers.Layer):
    def __init__(self, output_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)
        self.output_size = standardize_tuple(output_size, 2, "output_size")

    def call(self, inputs, *args, **kwargs):
        return adaptive_average_pooling_2d(inputs, self.output_size)

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.output_size[0],
            self.output_size[1],
            input_shape[3],
        )

    def get_config(self):
        config = super().get_config()
        config.update({"output_size": self.output_size})

        return config


@register_keras_serializable(package="SegMe>Common")
class MultiHeadAttentionPooling(layers.Layer):
    """Proposed in: https://arxiv.org/abs/2205.01917"""

    def __init__(self, heads, queries, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)
        self.heads = heads
        self.queries = queries

    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError(
                "Channel dimensions of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = InputSpec(min_ndim=3, axes={-1: self.channels})

        # noinspection PyAttributeOutsideInit
        self.query = self.add_weight(
            name="query", shape=(1, self.queries, self.channels)
        )

        # noinspection PyAttributeOutsideInit
        self.ln_q = Norm(
            policy="conv-ln1em5-relu", name="ln_q", dtype=self.dtype_policy
        )
        self.ln_q.build(self.query.shape)

        # noinspection PyAttributeOutsideInit
        self.ln_k = Norm(
            policy="conv-ln1em5-relu", name="ln_k", dtype=self.dtype_policy
        )
        self.ln_k.build(input_shape)

        # noinspection PyAttributeOutsideInit
        self.mhsa = layers.MultiHeadAttention(
            self.heads,
            self.channels // self.heads,
            name="mhsa",
            dtype=self.dtype_policy,
        )
        self.mhsa.build(
            self.query.shape, (input_shape[0], None, input_shape[-1])
        )

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        batch = ops.shape(inputs)[0]

        q = self.ln_q(self.query)
        q = ops.repeat(q, batch, axis=0)

        k = self.ln_k(inputs)
        k = ops.reshape(k, [batch, -1, self.channels])
        x = self.mhsa(q, k)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:1] + (self.queries,) + input_shape[-1:]

    def get_config(self):
        config = super().get_config()
        config.update({"heads": self.heads, "queries": self.queries})

        return config


@register_keras_serializable(package="SegMe>Common")
class SimPool(layers.Layer):
    """Proposed in: https://arxiv.org/abs/2309.06891"""

    def __init__(self, num_heads, qkv_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

        self.num_heads = num_heads
        self.qkv_bias = qkv_bias

    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        self.kv_norm = Norm(name="kv_norm", dtype=self.dtype_policy)
        self.kv_norm.build(input_shape)

        self.q_proj = Conv(
            self.channels,
            1,
            use_bias=self.qkv_bias,
            name="q_proj",
            dtype=self.dtype_policy,
        )
        self.q_proj.build((input_shape[0], 1, 1, input_shape[-1]))

        self.k_proj = Conv(
            self.channels,
            1,
            use_bias=self.qkv_bias,
            name="k_proj",
            dtype=self.dtype_policy,
        )
        self.k_proj.build(input_shape)

        self.scale = (self.channels // self.num_heads) ** -0.5

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        batch, height, width, _ = ops.shape(inputs)
        length = height * width

        q = ops.mean(inputs, axis=[1, 2], keepdims=True)
        q = self.q_proj(q)
        q = ops.reshape(
            q, [batch, self.num_heads, 1, self.channels // self.num_heads]
        )

        k = v = self.kv_norm(inputs)
        k = self.k_proj(k)
        k = ops.reshape(
            k, [batch, length, self.num_heads, self.channels // self.num_heads]
        )
        k = ops.transpose(k, [0, 2, 3, 1])
        v = ops.reshape(
            v, [batch, length, self.num_heads, self.channels // self.num_heads]
        )
        v = ops.transpose(v, [0, 2, 1, 3])

        attn = ops.matmul(q * self.scale, k)
        attn = ops.softmax(attn)

        outputs = ops.matmul(attn, v)
        outputs = ops.reshape(outputs, [batch, self.channels])

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:1] + input_shape[-1:]

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "qkv_bias": self.qkv_bias})

        return config

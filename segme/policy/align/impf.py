import numpy as np
import tensorflow as tf
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.convnormact import Conv
from segme.common.convnormact import ConvNormAct
from segme.common.impfunc import make_coords
from segme.common.impfunc import query_features
from segme.common.sequence import Sequence


@register_keras_serializable(package="SegMe>Policy>Align>LIIF")
class ImplicitFeatureAlignment(layers.Layer):
    """
    Proposed in "Learning Implicit Feature Alignment Function for Semantic
    Segmentation"
    https://arxiv.org/pdf/2206.08655.pdf
    """

    def __init__(self, filters=256, **kwargs):
        super().__init__(**kwargs)

        self.filters = filters

    def build(self, input_shape):
        self.channels = [shape[-1] for shape in input_shape]
        if None in self.channels:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = [
            InputSpec(ndim=4, axes={-1: c}) for c in self.channels
        ]

        self.posemb = []
        for _ in input_shape:
            pe = SpatialEncoding(dtype=self.dtype_policy)
            pe.build(input_shape[0][:-1] + (2,))
            self.posemb.append(pe)

        self.imnet = Sequence(
            [
                ConvNormAct(self.filters * 2, 1, dtype=self.dtype_policy),
                ConvNormAct(self.filters, 1, dtype=self.dtype_policy),
                Conv(self.filters, 1, dtype=self.dtype_policy),
            ],
            dtype=self.dtype_policy,
        )
        self.imnet.build(
            input_shape[0][:-1] + (sum(self.channels) + 26 * len(input_shape),)
        )

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        coords = make_coords(inputs[0], self.compute_dtype)

        contexts = []
        for i, feat in enumerate(inputs):
            columns = query_features(
                feat,
                coords,
                tf.identity,
                posnet=self.posemb[i],
                cells=None,
                feat_unfold=False,
                local_ensemble=False,
            )
            contexts.append(columns)
        contexts = tf.concat(contexts, axis=-1)

        outputs = self.imnet(contexts)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})

        return config


@register_keras_serializable(package="SegMe>Policy>Align>LIIF")
class SpatialEncoding(layers.Layer):
    """
    Proposed in "Learning Implicit Feature Alignment Function for Semantic
    Segmentation"
    https://arxiv.org/pdf/2206.08655.pdf
    """

    def __init__(self, units=24, sigma=6, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=2)

        self.units = units
        self.sigma = sigma

    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )
        self.input_spec = InputSpec(
            ndim=len(input_shape), axes={-1: self.channels}
        )

        self.embed_dim = max(1, self.units // 2 // self.channels)
        embed_dtype = tf.dtypes.as_dtype(self.dtype).as_numpy_dtype()
        embed_init = 2 ** np.linspace(0, self.sigma, self.embed_dim)
        embed_init = np.stack(
            [embed_init] + [np.zeros_like(embed_init)] * (self.channels - 1),
            axis=-1,
        )
        embed_init = np.concatenate(
            [np.roll(embed_init, i, axis=-1) for i in range(self.channels)],
            axis=0,
        )
        embed_init = embed_init.T.astype(embed_dtype)

        self.embedding = self.add_weight(
            name="embedding",
            shape=embed_init.shape,
            initializer=lambda shape, dtype=None: embed_init,
            trainable=True,
            dtype=self.dtype,
        )

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        embeddings = tf.matmul(inputs, self.embedding)
        outputs = tf.concat(
            [inputs, tf.sin(embeddings), tf.cos(embeddings)], axis=-1
        )

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.channels * (self.embed_dim * 2 + 1),)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "sigma": self.sigma})

        return config

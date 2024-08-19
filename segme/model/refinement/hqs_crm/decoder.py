import tensorflow as tf
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.convnormact import Conv
from segme.common.convnormact import ConvAct
from segme.common.convnormact import ConvNormAct
from segme.common.impfunc import query_features
from segme.common.resize import BilinearInterpolation
from segme.common.sequence import Sequence


@register_keras_serializable(package="SegMe>Model>Refinement>HqsCrm")
class Decoder(layers.Layer):
    def __init__(self, aspp_filters, aspp_drop, mlp_units, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=4) for _ in range(3)] + [
            InputSpec(ndim=4, axes={-1: 2})
        ]

        self.aspp_filters = aspp_filters
        self.aspp_drop = aspp_drop
        self.mlp_units = mlp_units

    def build(self, input_shape):
        self.resize = BilinearInterpolation(None, dtype=self.dtype_policy)

        self.aspp2 = ConvNormAct(
            self.aspp_filters[0], 1, dtype=self.dtype_policy
        )
        self.aspp2.build(input_shape[0])

        self.aspp4 = ConvNormAct(
            self.aspp_filters[1], 1, dtype=self.dtype_policy
        )
        self.aspp4.build(input_shape[1])

        self.aspp32 = ConvNormAct(
            self.aspp_filters[2], 1, dtype=self.dtype_policy
        )
        self.aspp32.build(input_shape[2])

        self.fuse = ConvNormAct(
            sum(self.aspp_filters), 3, dtype=self.dtype_policy
        )
        self.fuse.build(
            input_shape[0][:-1] + (sum(s[-1] for s in input_shape),)
        )

        self.drop = layers.Dropout(self.aspp_drop, dtype=self.dtype_policy)

        self.imnet = Sequence(
            [ConvAct(u, 1, dtype=self.dtype_policy) for u in self.mlp_units]
            + [Conv(1, 1)]
        )

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        feats2, feats4, feats32, coords = inputs

        aspp2 = self.aspp2(feats2)
        aspp2 = self.drop(aspp2)

        aspp4 = self.aspp4(feats4)
        aspp4 = self.drop(aspp4)
        aspp4 = self.resize([aspp4, aspp2])

        aspp32 = self.aspp32(feats32)
        aspp32 = self.drop(aspp32)
        aspp32 = self.resize([aspp32, aspp2])

        aspp = tf.concat([aspp2, aspp4, aspp32], axis=-1)
        aspp = self.fuse(aspp)
        aspp = self.drop(aspp)

        # Original implementation uses cells, but this is meaningless due to
        # input/output scale is constant
        outputs = query_features(
            aspp,
            coords,
            self.imnet,
            posnet=None,
            cells=None,
            feat_unfold=False,
            local_ensemble=True,
        )

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[-1][:-1] + (1,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mlp_units": self.mlp_units,
                "aspp_drop": self.aspp_drop,
                "aspp_filters": self.aspp_filters,
            }
        )

        return config

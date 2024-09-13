from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.convnormact import Conv
from segme.common.convnormact import ConvAct
from segme.common.impfunc import query_features
from segme.common.sequence import Sequence


@register_keras_serializable(package="SegMe>Model>Refinement>HqsCrm")
class Decoder(layers.Layer):
    def __init__(self, mlp_units, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4, axes={-1: 2})]
        self.mlp_units = mlp_units

    def build(self, input_shape):
        self.imnet = Sequence(
            [
                ConvAct(u, 1, dtype=self.dtype_policy, name=f"imnet_ca{i}")
                for i, u in enumerate(self.mlp_units)
            ]
            + [Conv(1, 1, name="imnet_proj")],
            name="imnet",
        )
        self.imnet.build(input_shape[1][:-1] + (input_shape[0][-1] + 2,))

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        aspp, coord = inputs

        # Original implementation uses cells, but this is meaningless due to
        # input/output scale is constant
        outputs = query_features(
            aspp,
            coord,
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
        config.update({"mlp_units": self.mlp_units})

        return config

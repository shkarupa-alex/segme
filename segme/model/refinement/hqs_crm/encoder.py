from keras.src import layers
from keras.src import models
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.policy import bbpol
from segme.policy.backbone.utils import patch_config


@register_keras_serializable(package="SegMe>Model>Refinement>HqsCrm")
class Encoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4, axes={-1: 4})

    def build(self, input_shape):
        base_model = bbpol.BACKBONES.new(
            "resnet_rs_50_s8", "imagenet", 3, [2, 4, 32]
        )
        base_config = base_model.get_config()
        base_weights = base_model.get_weights()

        ext_config = base_config
        ext_config["layers"][0]["config"]["batch_shape"] = (None, None, None, 4)
        ext_config["layers"][1]["build_config"]["input_shape"] = (
            None,
            None,
            None,
            4,
        )
        ext_config["layers"][1]["inbound_nodes"][0]["args"][0]["config"][
            "shape"
        ] = (None, None, None, 4)
        ext_config["layers"][2]["build_config"]["input_shape"] = (
            None,
            None,
            None,
            4,
        )
        ext_config["layers"][2]["inbound_nodes"][0]["args"][0]["config"][
            "shape"
        ] = (None, None, None, 4)
        ext_config["layers"][3]["build_config"]["input_shape"] = (
            None,
            None,
            None,
            4,
        )
        ext_config["layers"][3]["inbound_nodes"][0]["args"][0]["config"][
            "shape"
        ] = (None, None, None, 4)
        ext_config = patch_config(
            ext_config, [2], "mean", lambda old: old + [0.408]
        )
        ext_config = patch_config(
            ext_config, [2], "variance", lambda old: old + [0.492**2]
        )
        ext_model = models.Model.from_config(ext_config)

        ext_weights = []
        for base_weight, ext_weight in zip(
            base_weights, ext_model.get_weights()
        ):
            if base_weight.shape != ext_weight.shape:
                if (
                    base_weight.shape[:2] + base_weight.shape[3:]
                    != ext_weight.shape[:2] + ext_weight.shape[3:]
                ):
                    raise ValueError("Unexpected weight shape")

                ext_weight[:, :, : base_weight.shape[2]] = base_weight
                ext_weights.append(ext_weight)
            else:
                ext_weights.append(base_weight)

        ext_model.set_weights(ext_weights)
        ext_model.trainable = True

        self.backbone = ext_model

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.backbone(inputs)

    def compute_output_shape(self, input_shape):
        return self.backbone.compute_output_shape(input_shape)

from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.policy import bbpol
from segme.policy.backbone.utils import patch_config


@register_keras_serializable(package='SegMe>Model>Matting>FBAMatting')
class Encoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: 11})

    @shape_type_conversion
    def build(self, input_shape):
        base_model = bbpol.BACKBONES.new('resnet_rs_50_s8', 'imagenet', 3, [2, 4, 32])
        base_config = base_model.get_config()
        base_weights = base_model.get_weights()

        ext_config = patch_config(base_config, [0], 'batch_input_shape', lambda old: old[:-1] + (11,))
        ext_config = patch_config(ext_config, [2], 'mean', lambda old: old + [
            0.187, 0.338, 0.207, 0.267, 0.338, 0.369, 0.448, 0.527])
        ext_config = patch_config(ext_config, [2], 'variance', lambda old: old + [
            0.389 ** 2, 0.473 ** 2, 0.397 ** 2, 0.416 ** 2, 0.428 ** 2, 0.473 ** 2, 0.468 ** 2, 0.453 ** 2])
        ext_model = models.Model.from_config(ext_config)

        ext_weights = []
        for base_weight, ext_weight in zip(base_weights, ext_model.get_weights()):
            if base_weight.shape != ext_weight.shape:
                if base_weight.shape[:2] + base_weight.shape[3:] != ext_weight.shape[:2] + ext_weight.shape[3:]:
                    raise ValueError('Unexpected weight shape')

                ext_weight[:, :, :base_weight.shape[2]] = base_weight
                ext_weights.append(ext_weight)
            else:
                ext_weights.append(base_weight)

        ext_model.set_weights(ext_weights)
        ext_model.trainable = True

        self.backbone = ext_model

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.backbone(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.backbone.compute_output_shape(input_shape)

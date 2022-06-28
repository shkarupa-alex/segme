import tensorflow as tf
from keras import layers, models
from keras.applications import resnet_rs
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...backbone.utils import patch_config


@register_keras_serializable(package='SegMe>HqsCrm')
class Encoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # image
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8')  # mask
        ]

    @shape_type_conversion
    def build(self, input_shape):
        base_model = self._default_model()

        base_config = base_model.get_config()
        ext_config = self._extend_config(base_config)
        ext_model = models.Model.from_config(ext_config)

        base_weights = base_model.get_weights()
        ext_weights = ext_model.get_weights()
        ext_weights = self._extend_weights(base_weights, ext_weights)
        ext_model.set_weights(ext_weights)

        self.backbone = ext_model

        super().build(input_shape)

    def _default_model(self):
        input_image = layers.Input(name='image', shape=(None, None, 3), dtype=self.compute_dtype)
        base_model = resnet_rs.ResNetRS50(input_tensor=input_image, include_top=False, weights='imagenet')

        end_points = [12, 63, 270]
        out_layers = [base_model.get_layer(index=i).output for i in end_points]

        return models.Model(inputs=input_image, outputs=out_layers)

    def _extend_config(self, config):
        # Patch input shape
        config = patch_config(config, [0], 'batch_input_shape', lambda old: old[:-1] + (4,))
        config = patch_config(config, [2], 'mean', lambda old: old + [0.182015])
        config = patch_config(config, [2], 'variance', lambda old: old + [0.148886])

        # Group 4: first block
        config = patch_config(config, [131], 'padding', ((0, 0), (0, 0)))
        config = patch_config(config, [132], 'strides', (1, 1))
        config = patch_config(config, [132], 'padding', 'same')

        # Group 4: residual
        config = patch_config(config, [140], 'pool_size', (1, 1))
        config = patch_config(config, [140], 'strides', (1, 1))

        # Group 4: rest blocks
        config = patch_config(config, [150], 'dilation_rate', (2, 2))
        config = patch_config(config, [165], 'dilation_rate', (2, 2))
        config = patch_config(config, [180], 'dilation_rate', (2, 2))
        config = patch_config(config, [195], 'dilation_rate', (2, 2))
        config = patch_config(config, [210], 'dilation_rate', (2, 2))

        # Group 5: first block
        config = patch_config(config, [225], 'padding', ((0, 0), (0, 0)))
        config = patch_config(config, [226], 'strides', (1, 1))
        config = patch_config(config, [226], 'padding', 'same')

        # Group 5: residual
        config = patch_config(config, [234], 'pool_size', (1, 1))
        config = patch_config(config, [234], 'strides', (1, 1))

        # Group 5: rest blocks
        config = patch_config(config, [244], 'dilation_rate', (4, 4))
        config = patch_config(config, [259], 'dilation_rate', (4, 4))

        return config

    def _extend_weights(self, base_weights, ext_weights):
        weights = []
        base_idx = 0
        for ext_weight in ext_weights:
            if base_idx < len(base_weights) and ext_weight.shape == base_weights[base_idx].shape:
                weights.append(base_weights[base_idx])
                base_idx += 1
            elif base_idx < len(base_weights) and \
                    (3, 3, 4, 32) == ext_weight.shape and (3, 3, 3, 32) == base_weights[base_idx].shape:
                ext_weight[:, :, :3, :] = base_weights[base_idx][base_idx]
                weights.append(ext_weight)
                base_idx += 1
            elif 2 == len(ext_weight.shape):
                weights.append(ext_weight)
            else:
                raise ValueError('Unable to transfer weight')

        return weights

    def call(self, inputs, **kwargs):
        images, masks = inputs
        combos = tf.concat([images, masks], axis=-1)
        combos = tf.cast(combos, self.compute_dtype)

        return self.backbone(combos)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.backbone.compute_output_shape(input_shape[0][:-1] + (4,))

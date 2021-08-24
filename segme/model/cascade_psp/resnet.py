import numpy as np
from keras import Model, layers
from keras.applications import resnet
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='SegMe>CascadePSP')
class ResNet50(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: 6})

    @shape_type_conversion
    def build(self, input_shape):
        base_model = self._default_resnet50()

        ext_config = self._extend_config(base_model.get_config())
        ext_weights = self._extend_weights(base_model.get_weights())

        ext_model = Model.from_config(ext_config)
        ext_model.set_weights(ext_weights)

        self.backbone = ext_model

        super().build(input_shape)

    def _default_resnet50(self):
        input_image = layers.Input(name='image', shape=(None, None, 3), dtype=self.compute_dtype)
        base_model = resnet.ResNet50(input_tensor=input_image, include_top=False, weights='imagenet')

        end_points = ['conv1_relu', 'conv2_block3_out', 'conv5_block3_out']
        out_layers = [base_model.get_layer(name=name_idx).output for name_idx in end_points]

        return Model(inputs=input_image, outputs=out_layers)

    def _extend_config(self, base_config):
        config = dict(base_config)
        config = self._patch_config(config, 0, 'batch_input_shape', lambda old: old[:-1] + (6,))
        config = self._patch_config(config, 'conv4_block1_0_conv', 'strides', lambda _: (1, 1))
        config = self._patch_config(config, 'conv4_block1_1_conv', 'strides', lambda _: (1, 1))

        config = self._patch_config(config, 'conv4_block2_2_conv', 'dilation_rate', lambda _: (2, 2))
        config = self._patch_config(config, 'conv4_block3_2_conv', 'dilation_rate', lambda _: (2, 2))
        config = self._patch_config(config, 'conv4_block4_2_conv', 'dilation_rate', lambda _: (2, 2))
        config = self._patch_config(config, 'conv4_block5_2_conv', 'dilation_rate', lambda _: (2, 2))
        config = self._patch_config(config, 'conv4_block6_2_conv', 'dilation_rate', lambda _: (2, 2))

        config = self._patch_config(config, 'conv5_block1_0_conv', 'strides', lambda _: (1, 1))
        config = self._patch_config(config, 'conv5_block1_1_conv', 'strides', lambda _: (1, 1))

        config = self._patch_config(config, 'conv5_block2_2_conv', 'dilation_rate', lambda _: (4, 4))
        config = self._patch_config(config, 'conv5_block3_2_conv', 'dilation_rate', lambda _: (4, 4))

        return config

    def _patch_config(self, config, layer, param, patch):
        for i in range(len(config['layers'])):
            if isinstance(layer, int):
                if i != layer:
                    continue
            elif config['layers'][i]['name'] != layer:
                continue

            if param not in config['layers'][i]['config']:
                raise ValueError('Parameter {} does not exist in layer {}'.format(layer, param))

            config['layers'][i]['config'][param] = patch(config['layers'][i]['config'][param])

            return config

        raise ValueError('Layer {} not found'.format(layer))

    def _extend_weights(self, weights):
        return [w if w.shape != (7, 7, 3, 64) else np.concatenate([w, np.zeros_like(w)], axis=2) for w in weights]

    def call(self, inputs, **kwargs):
        return self.backbone(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.backbone.compute_output_shape(input_shape)

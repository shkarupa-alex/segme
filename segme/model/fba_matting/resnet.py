import numpy as np
from tensorflow.keras import Model, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from ...backbone.port.big_transfer import bit
from ...backbone.utils import get_layer


@utils.register_keras_serializable(package='SegMe>FBAMatting')
class ResNet50(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_channels = 11
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: self.input_channels})

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
        base_model = bit.BiT_M_R50x1(input_tensor=input_image, include_top=False, weights='imagenet')

        end_points = ['standardized_conv2d', 'block1_out', 'block4_out']  # x2, x4, x32
        out_layers = [get_layer(base_model, name_idx) for name_idx in end_points]

        return Model(inputs=input_image, outputs=out_layers)

    def _extend_config(self, base_config):
        config = dict(base_config)
        config = self._patch_config(config, [0], 'batch_input_shape', lambda old: old[:-1] + (self.input_channels,))

        config = self._patch_config(config, ['block3', 'unit01'], 'stride', 1)
        config = self._patch_config(config, ['block3', 'unit02'], 'dilation', 2)
        config = self._patch_config(config, ['block3', 'unit03'], 'dilation', 2)
        config = self._patch_config(config, ['block3', 'unit04'], 'dilation', 2)
        config = self._patch_config(config, ['block3', 'unit05'], 'dilation', 2)
        config = self._patch_config(config, ['block3', 'unit06'], 'dilation', 2)

        config = self._patch_config(config, ['block4', 'unit01'], 'stride', 1)
        config = self._patch_config(config, ['block4', 'unit01'], 'dilation', 2)
        config = self._patch_config(config, ['block4', 'unit02'], 'dilation', 4)
        config = self._patch_config(config, ['block4', 'unit03'], 'dilation', 4)

        return config

    def _patch_config(self, config, path, param, patch):
        if 'layers' not in config:
            raise ValueError('Can\'t find layers in config {}'.format(config))

        head, tail = path[0], path[1:]
        for i in range(len(config['layers'])):
            found = isinstance(head, int) and i == head or config['layers'][i]['config'].get('name', None) == head
            if not found:
                continue

            if tail:
                config['layers'][i]['config'] = self._patch_config(config['layers'][i]['config'], tail, param, patch)
            elif param not in config['layers'][i]['config']:
                raise ValueError('Parameter {} not found in layer {}'.format(param, head))
            else:
                patched = patch if not callable(patch) else patch(config['layers'][i]['config'][param])
                config['layers'][i]['config'][param] = patched

            return config

        raise ValueError('Layer {} not found'.format(head))

    def _extend_weights(self, weights):
        zeros_shape = (7, 7, self.input_channels - 3, 64)
        return [w if w.shape != (7, 7, 3, 64) else np.concatenate([w, np.zeros(zeros_shape)], axis=2) for w in weights]

    def call(self, inputs, **kwargs):
        return self.backbone(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.backbone.compute_output_shape(input_shape)

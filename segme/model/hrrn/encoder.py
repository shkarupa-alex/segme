import tensorflow as tf
from keras import layers, models
from keras.applications import resnet_rs
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tensorflow_addons.layers import SpectralNormalization  # Required to initialize custom layer


@register_keras_serializable(package='SegMe>HRRN')
class Encoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # image
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8')  # trimap
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

        end_points = [2, 12, 63, 127, 221, 270]
        out_layers = [base_model.get_layer(index=i).output for i in end_points]

        return models.Model(inputs=input_image, outputs=out_layers)

    def _extend_config(self, config):
        # Patch input shape
        config = self._patch_config(config, 0, 'batch_input_shape', lambda old: old[:-1] + (6,))
        config = self._patch_config(config, 2, 'mean', lambda old: old + [0.782468, 0.071937, 0.145596])
        config = self._patch_config(config, 2, 'variance', lambda old: old + [0.199430, 0.076148, 0.155288])

        # # Restore default BatchNormalization parameters
        # for i, layer in enumerate(config['layers']):
        #     if 'BatchNormalization' != layer['class_name']:
        #         continue
        #
        #     config['layers'][i]['config']['momentum'] = 0.99

        # Apply SpectralNormalization to all Conv2D layers
        for i, layer in enumerate(config['layers']):
            if 'Conv2D' != layer['class_name']:
                continue

            config['layers'][i]['class_name'] = 'Addons>SpectralNormalization'
            config['layers'][i]['config'] = {
                'name': layer['config']['name'],
                'trainable': True,
                'dtype': layer['config']['dtype'],
                'layer': {
                    'class_name': 'Conv2D',
                    'config': layer['config']
                },
                'power_iterations': 1
            }
            config['layers'][i]['config']['layer']['config']['name'] += '_wrapped'

        # config = self._patch_config(config, 'conv4_block1_0_conv', 'strides', lambda _: (1, 1))
        # config = self._patch_config(config, 'conv4_block1_1_conv', 'strides', lambda _: (1, 1))
        #
        # config = self._patch_config(config, 'conv4_block2_2_conv', 'dilation_rate', lambda _: (2, 2))
        # config = self._patch_config(config, 'conv4_block3_2_conv', 'dilation_rate', lambda _: (2, 2))
        # config = self._patch_config(config, 'conv4_block4_2_conv', 'dilation_rate', lambda _: (2, 2))
        # config = self._patch_config(config, 'conv4_block5_2_conv', 'dilation_rate', lambda _: (2, 2))
        # config = self._patch_config(config, 'conv4_block6_2_conv', 'dilation_rate', lambda _: (2, 2))
        #
        # config = self._patch_config(config, 'conv5_block1_0_conv', 'strides', lambda _: (1, 1))
        # config = self._patch_config(config, 'conv5_block1_1_conv', 'strides', lambda _: (1, 1))
        #
        # config = self._patch_config(config, 'conv5_block2_2_conv', 'dilation_rate', lambda _: (4, 4))
        # config = self._patch_config(config, 'conv5_block3_2_conv', 'dilation_rate', lambda _: (4, 4))

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

    def _extend_weights(self, base_weights, ext_weights):
        weights = []
        base_idx = 0
        for ext_weight in ext_weights:
            if base_idx < len(base_weights) and ext_weight.shape == base_weights[base_idx].shape:
                weights.append(base_weights[base_idx])
                base_idx += 1
            elif base_idx < len(base_weights) and \
                    (3, 3, 6, 32) == ext_weight.shape and (3, 3, 3, 32) == base_weights[base_idx].shape:
                ext_weight[:, :, :3, :] = base_weights[base_idx][base_idx]
                weights.append(ext_weight)
                base_idx += 1
            elif 2 == len(ext_weight.shape):
                weights.append(ext_weight)
            else:
                raise ValueError('Unable to transfer weight')

        return weights

    def call(self, inputs, **kwargs):
        images, trimaps = inputs
        trimaps = tf.one_hot(trimaps[..., 0] // 86, 3, dtype='uint8') * 255

        combos = tf.concat([images, trimaps], axis=-1)
        combos = tf.cast(combos, self.compute_dtype)

        return self.backbone(combos)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.backbone.compute_output_shape(input_shape[0][:-1] + (6,))

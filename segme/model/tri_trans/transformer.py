import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from scipy import ndimage
from vit_keras import vit
from ...backbone.utils import patch_config
from ...common import ConvNormRelu


@register_keras_serializable(package='SegMe>TriTrans')
class VisionTransformer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

    @shape_type_conversion
    def build(self, input_shape):
        width, height, channels = input_shape[1:]
        if width is None or height is None or channels is None:
            raise ValueError('Width, height and channel dimensions of the inputs should be defined. Found `None`.')
        if width != height:
            raise ValueError('Only square images supported. Provided: {}.'.format(input_shape))

        self.input_spec = layers.InputSpec(ndim=4, axes={1: width, 2: height, 3: channels})

        base_model = vit.vit_b16(
            image_size=(224, 224), pretrained=True, include_top=False, pretrained_top=False, weights='imagenet21k')
        outputs = base_model.get_layer(name='Transformer/encoder_norm').output
        outputs = layers.Lambda(lambda v: v[:, 1:, ...], name='ExtractFeatures')(outputs)
        base_model = models.Model(inputs=base_model.inputs, outputs=outputs)

        base_config = base_model.get_config()
        vit_config = patch_config(base_config, [0], 'batch_input_shape', (None, width, height, channels))
        vit_config = patch_config(vit_config, ['embedding'], 'kernel_size', (1, 1))
        vit_config = patch_config(vit_config, ['embedding'], 'strides', (1, 1))
        vit_config = patch_config(vit_config, [2], 'target_shape', lambda old: (width * height,) + old[-1:])
        vit_model = models.Model.from_config(vit_config)

        def _ext_weight(wb, wv):
            if wb.shape == wv.shape:
                return wb

            if (16, 16, 3) == wb.shape[:3] and (1, 1) == wv.shape[:2]:
                # embedding
                # will be trained from scratch
                return wv
            if 3 == len(wb.shape) and wb.shape[0] == wv.shape[0] == 1 and wb.shape[2] == wv.shape[2]:
                # posembed_input
                token, grid = wb[0, :1], wb[0, 1:]
                sin = int(np.sqrt(grid.shape[0]))
                zoom = (width / sin, height / sin, 1)
                grid = grid.reshape(sin, sin, -1)
                grid = ndimage.zoom(grid, zoom, order=1).reshape(width * height, -1)
                combo = np.concatenate([token, grid], axis=0)[None]
                assert combo.shape == wv.shape

                return combo

            return wb  # will raise error if something changes

        base_weights = base_model.get_weights()
        vit_weights = [_ext_weight(wb, wv) for wb, wv in zip(base_weights, vit_model.get_weights())]
        vit_model.set_weights(vit_weights)
        self.vit = vit_model

        self.decoder = DecoderCup(channels)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.vit(inputs)
        outputs = self.decoder(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = self.vit.compute_output_shape(input_shape)
        output_shape = self.decoder.compute_output_shape(output_shape)

        return output_shape


@register_keras_serializable(package='SegMe>TriTrans')
class DecoderCup(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

        self.filters = filters

    @shape_type_conversion
    def build(self, input_shape):
        width_height, channels = input_shape[1:]
        if width_height is None or channels is None:
            raise ValueError('Width/height and channel dimensions of the inputs should be defined. Found `None`.')
        if width_height != int(width_height ** 0.5) ** 2:
            raise ValueError('Provided input can\'t be reshaped to square image.')

        self.input_spec = layers.InputSpec(ndim=3, axes={1: width_height, 2: channels})

        self.width_height = int(width_height ** 0.5)
        self.channels = channels

        self.conv = ConvNormRelu(self.filters, kernel_size=3, padding='same')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.reshape(inputs, [-1, self.width_height, self.width_height, self.channels])
        outputs = self.conv(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        width_height = int(input_shape[1] ** 0.5)

        return input_shape[:1] + (width_height, width_height, self.filters)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})

        return config

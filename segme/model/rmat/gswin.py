import numpy as np
import tensorflow as tf
import tfswin
from keras import models, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfswin.norm import LayerNorm


@register_keras_serializable(package='SegMe>RMat')
class GidedSwin(layers.Layer):
    def __init__(self, arch, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),
            layers.InputSpec(ndim=4, dtype='uint8')]

        self.arch = arch

    @shape_type_conversion
    def build(self, input_shape):
        arch2model = {
            'tiny_224': tfswin.SwinTransformerTiny224,
            'small_224': tfswin.SwinTransformerSmall224,
            'base_224': tfswin.SwinTransformerBase224,
            'base_384': tfswin.SwinTransformerBase384,
            'large_224': tfswin.SwinTransformerLarge224,
            'large_384': tfswin.SwinTransformerLarge384
        }
        if self.arch not in arch2model:
            raise ValueError('Unsupported Swin Transformer architecture.')

        self.channels = [input_shape[i][-1] for i in range(2)]
        if None in self.channels:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        self.channels = sum(self.channels)

        base_model = arch2model[self.arch]()
        base_config = base_model.get_config()

        ext_config = dict(base_config)
        ext_config['layers'][0]['config']['batch_input_shape'] = (None, None, None, self.channels)

        ext_model = models.Model.from_config(ext_config)
        ext_weights = [
            wb if wb.shape == we.shape else np.concatenate([wb, we[:, :, 3:, :]], axis=2)
            for wb, we in zip(base_model.get_weights(), ext_model.get_weights())]
        ext_model.set_weights(ext_weights)

        end_points = ['layers.0', 'layers.1', 'layers.2', 'layers.3']
        ext_outputs = [ext_model.get_layer(name=name).output for name in end_points]
        self.backbone = models.Model(
            inputs=ext_model.inputs,
            outputs=ext_outputs)

        self.norm4 = LayerNorm()
        self.norm8 = LayerNorm()
        self.norm16 = LayerNorm()
        self.norm32 = LayerNorm()

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        image, guide = inputs
        image = tfswin.preprocess_input(image)
        image = tf.cast(image, self.compute_dtype)
        guide = tf.cast(guide, self.compute_dtype)
        guide = guide / (255. * 0.289) - 0.5 / 0.289

        combo = tf.concat([image, guide], axis=-1)
        feat4, feat8, feat16, feat32 = self.backbone(combo)

        feat4 = self.norm4(feat4)
        feat8 = self.norm8(feat8)
        feat16 = self.norm16(feat16)
        feat32 = self.norm32(feat32)

        return feat4, feat8, feat16, feat32

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        combo_shape = input_shape[0][:-1] + (self.channels,)

        return self.backbone.compute_output_shape(combo_shape)

    def get_config(self):
        config = super().get_config()
        config.update({'arch': self.arch})

        return config

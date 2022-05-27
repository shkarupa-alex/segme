import tensorflow as tf
import tfswin
from keras import layers, models
from keras.applications import imagenet_utils
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tensorflow_addons import layers as addon_layers
from ...common import SameConv, ResizeByScale


@register_keras_serializable(package='SegMe>MatteFormer')
class Encoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # image
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8')  # trimap
        ]

    def model(self, channels, init):
        input_shape = (None, None, channels)
        input_image = layers.Input(name='image', shape=input_shape, dtype=self.compute_dtype)
        base_model = tfswin.model.SwinTransformerTiny224(input_tensor=input_image, weights=init)

        end_points = [ep for ep in ['patch_embed', 'layers.0', 'layers.1', 'layers.2', 'layers.3']]
        out_layers = [base_model.get_layer(name=name).output for name in end_points]

        down_stack = models.Model(inputs=input_image, outputs=out_layers)

        return down_stack

    @shape_type_conversion
    def build(self, input_shape):
        base_model = self.model(3, 'imagenet')
        ext_model = self.model(6, None)

        ext_weights = []
        for base_weight, ext_weight in zip(base_model.get_weights(), ext_model.get_weights()):
            if base_weight.shape != ext_weight.shape:
                ext_weight[:, :, :3, :] = base_weight
            ext_weights.append(ext_weight)
        ext_model.set_weights(ext_weights)

        self.backbone = ext_model
        self.shortcuts = [
            models.Sequential([
                addon_layers.SpectralNormalization(SameConv(filters, 3, use_bias=False)),
                layers.ReLU(),
                layers.BatchNormalization(),
                addon_layers.SpectralNormalization(SameConv(filters, 3, use_bias=False)),
                layers.ReLU(),
                layers.BatchNormalization()])
            for filters in [32, 32, 64, 128, 256, 512]]
        self.up2 = ResizeByScale(2)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        images, trimaps = inputs

        images = tf.cast(images, self.compute_dtype)
        images = imagenet_utils.preprocess_input(images, mode='torch')

        frgmap, bgrmap = trimaps < 85, trimaps > 170
        trimaps = tf.concat([frgmap, ~(frgmap | bgrmap), bgrmap], axis=-1)
        trimaps = tf.cast(trimaps, self.compute_dtype)
        # TODO: for v2
        # trimaps = (trimaps - .5) / .5

        outputs = tf.concat([images, trimaps], axis=-1)
        features = [self.shortcuts[0](outputs)]

        outputs = self.backbone(outputs)
        features.append(self.shortcuts[1](self.up2(outputs[0])))

        for out, short in zip(outputs[1:], self.shortcuts[2:]):
            features.append(short(out))

        return features

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = self.backbone.compute_output_shape(input_shape[0][:-1] + (6,))

        return [self.shortcuts[0].compute_output_shape(input_shape[0]),
                self.up2.compute_output_shape(output_shape[0])] + output_shape[1:]

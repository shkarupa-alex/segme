import tensorflow as tf
import tfswin
from keras import layers, models
from keras.applications import imagenet_utils
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvAct, Norm
from segme.common.sequent import Sequential
from segme.common.interrough import BilinearInterpolation
from segme.policy import cnapol


@register_keras_serializable(package='SegMe>Model>Matting>MatteFormer')
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
                # Single case: patch embedding
                ext_weight[:, :, :3, :] = base_weight
            ext_weights.append(ext_weight)
        ext_model.set_weights(ext_weights)

        self.backbone = ext_model

        with cnapol.policy_scope('snconv-bn-relu'):
            self.shortcuts = [
                Sequential([
                    ConvAct(filters, 3, use_bias=False), Norm(),
                    ConvAct(filters, 3, use_bias=False), Norm()])
                for filters in [32, 32, 64, 128, 256, 512]]
        self.up2 = BilinearInterpolation(2)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        images, trimaps = inputs

        images = tf.cast(images, self.compute_dtype)
        images = imagenet_utils.preprocess_input(images, mode='torch')

        trimaps = tf.one_hot(trimaps[..., 0] // 86, 3, dtype=self.compute_dtype)
        trimaps = tf.nn.bias_add(trimaps, tf.cast([-0.090, -0.860, -0.050], self.compute_dtype))
        trimaps /= tf.cast([0.286, 0.347, 0.217], self.compute_dtype)

        outputs = tf.concat([images, trimaps], axis=-1)
        features = [self.shortcuts[0](outputs)]

        outputs = self.backbone(outputs)
        features.append(self.shortcuts[1](self.up2(outputs[0])))

        for short, out in zip(self.shortcuts[2:], outputs[1:]):
            features.append(short(out))

        return features

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        combined_shape = input_shape[0][:-1] + (6,)

        output_shape = [self.shortcuts[0].compute_output_shape(combined_shape)]

        backbone_shape = self.backbone.compute_output_shape(combined_shape)
        output_shape.append(self.shortcuts[1].compute_output_shape(
            self.up2.compute_output_shape(backbone_shape[0])))

        for bone, short in zip(backbone_shape[1:], self.shortcuts[2:]):
            output_shape.append(short.compute_output_shape(bone))

        return output_shape

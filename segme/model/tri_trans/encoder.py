import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.applications import resnet
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from ...backbone.utils import patch_config
from .enhance import CASAEnhance


@register_keras_serializable(package='SegMe>TriTrans')
class MMFusionEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # rgb
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint16')  # depth
        ]

    @shape_type_conversion
    def build(self, input_shape):
        # Default ResNet50
        base_input = layers.Input(name='image', shape=(None, None, 3), dtype=self.compute_dtype)
        base_model = resnet.ResNet50(input_tensor=base_input, include_top=False, weights='imagenet')

        # ResNet50 adapted for depth input
        depth_config = base_model.get_config()
        depth_config = patch_config(depth_config, [0], 'batch_input_shape', lambda old: old[:-1] + (1,))
        depth_model = models.Model.from_config(depth_config)

        base_weights = base_model.get_weights()
        depth_weights = [w if w.shape != (7, 7, 3, 64) else np.sum(w, axis=2, keepdims=True) for w in base_weights]
        depth_model.set_weights(depth_weights)

        end_points = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
        out_layers = [depth_model.get_layer(name=name_idx).output for name_idx in end_points]
        self.depth_bone = models.Model(inputs=depth_model.inputs, outputs=out_layers)

        # 6 + 60 + 78 + 114 + 60
        # ResNet50 sliced by feature layers
        self.rgb_bone2 = models.Model(inputs=base_model.inputs, outputs=base_model.get_layer(name='conv1_relu').output)
        self.rgb_bone2.set_weights(base_weights[0:6])

        rgb_bone4_inp = layers.Input(name='bone2_output', shape=(None, None, 64), dtype=self.compute_dtype)
        rgb_bone4_out = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(rgb_bone4_inp)
        rgb_bone4_out = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(rgb_bone4_out)
        rgb_bone4_out = resnet.stack1(rgb_bone4_out, 64, 3, stride1=1, name='conv2')
        self.rgb_bone4 = models.Model(inputs=rgb_bone4_inp, outputs=rgb_bone4_out)
        self.rgb_bone4.set_weights(base_weights[6:66])

        rgb_bone8_inp = layers.Input(name='bone4_output', shape=(None, None, 4 * 64), dtype=self.compute_dtype)
        rgb_bone8_out = resnet.stack1(rgb_bone8_inp, 128, 4, name='conv3')
        self.rgb_bone8 = models.Model(inputs=rgb_bone8_inp, outputs=rgb_bone8_out)
        self.rgb_bone8.set_weights(base_weights[66:144])

        rgb_bone16_inp = layers.Input(name='bone8_output', shape=(None, None, 4 * 128), dtype=self.compute_dtype)
        rgb_bone16_out = resnet.stack1(rgb_bone16_inp, 256, 6, name='conv4')
        self.rgb_bone16 = models.Model(inputs=rgb_bone16_inp, outputs=rgb_bone16_out)
        self.rgb_bone16.set_weights(base_weights[144:258])

        rgb_bone32_inp = layers.Input(name='bone16_output', shape=(None, None, 4 * 256), dtype=self.compute_dtype)
        rgb_bone32_out = resnet.stack1(rgb_bone32_inp, 512, 3, name='conv5')
        self.rgb_bone32 = models.Model(inputs=rgb_bone32_inp, outputs=rgb_bone32_out)
        self.rgb_bone32.set_weights(base_weights[258:318])

        self.enhance2 = CASAEnhance()
        self.enhance4 = CASAEnhance()
        self.enhance8 = CASAEnhance()
        self.enhance16 = CASAEnhance()
        self.enhance32 = CASAEnhance()

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        rgb, depth = inputs

        rgb = tf.cast(rgb, self.compute_dtype)
        rgb = rgb[..., ::-1]  # 'RGB'->'BGR'
        rgb = tf.nn.bias_add(rgb, [-103.939, -116.779, -123.680])

        depth = tf.cast(depth, self.compute_dtype)
        min_depth = tf.reduce_min(depth, axis=[1, 2], keepdims=True)  # TODO: correlate with augmentations
        max_depth = tf.reduce_max(depth, axis=[1, 2], keepdims=True)
        depth = 255. * (depth - min_depth) / (max_depth - min_depth)
        depth = tf.nn.bias_add(depth, [-114.799])

        depth_feats2, depth_feats4, depth_feats8, depth_feats16, depth_feats32 = self.depth_bone(depth)

        rgb_feats2 = self.rgb_bone2(rgb)
        rgb_feats2 += self.enhance2([rgb_feats2, depth_feats2])

        rgb_feats4 = self.rgb_bone4(rgb_feats2)
        rgb_feats4 += self.enhance4([rgb_feats4, depth_feats4])

        rgb_feats8 = self.rgb_bone8(rgb_feats4)
        rgb_feats8 += self.enhance8([rgb_feats8, depth_feats8])

        rgb_feats16 = self.rgb_bone16(rgb_feats8)
        rgb_feats16 += self.enhance16([rgb_feats16, depth_feats16])

        rgb_feats32 = self.rgb_bone32(rgb_feats16)
        rgb_feats32 += self.enhance32([rgb_feats32, depth_feats32])

        return rgb_feats2, rgb_feats4, rgb_feats8, rgb_feats16, rgb_feats32

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.depth_bone.compute_output_shape(input_shape[1])

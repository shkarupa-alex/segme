import tensorflow as tf
from keras import layers, models, utils
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .decoder import Decoder
from .distance import Distance
from .fusion import Fusion
from .encoder import Encoder
from .twomap import Twomap


@register_keras_serializable(package='SegMe>FBAMatting')
class FBAMatting(layers.Layer):
    def __init__(self, bone_arch, bone_init, pool_scales, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # image
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8'),  # trimap
        ]
        self.pool_scales = pool_scales
        self.bone_arch = bone_arch
        self.bone_init = bone_init

    def build(self, input_shape):
        self.twomap = Twomap()
        self.distance = Distance()
        self.encoder = Encoder(self.bone_arch, self.bone_init)
        self.decoder = Decoder(self.pool_scales)
        self.fusion = Fusion(dtype='float32')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        image, trimap = inputs  # TODO: scale to multiple of 8?
        twomap = self.twomap(trimap)  # [0; 1]
        distance = self.distance(twomap)  # [0; 1]

        # Rescale twomap and distance to match preprocessed image
        featraw = layers.concatenate([
            image,
            tf.cast(twomap * 255., 'uint8'),  # Same scale as  image
            tf.cast(tf.round(distance * 255.), 'uint8')  # Same scale as  image
        ], axis=-1)
        feats2, feats4, feats32 = self.encoder(featraw)

        imscal = tf.cast(image, self.compute_dtype) / 255.  # Same scale as twomap, alpha, foreground and background
        alfgbg = self.decoder([feats2, feats4, feats32, imscal, twomap])

        alpha, foreground, background = self.fusion([imscal, alfgbg])  # TODO: try twice?
        alpha = tf.round(alpha * 255.)  # TODO: cast?
        foreground = tf.round(foreground * 255.)
        background = tf.round(background * 255.)

        return alfgbg * 255., alpha, foreground, background

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        base_shape = input_shape[0][:-1]

        return base_shape + (7,), base_shape + (1,), base_shape + (3,), base_shape + (3,)

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)
        alfgbg_signature = tf.TensorSpec(dtype='float32', shape=outptut_signature[0].shape)

        return (alfgbg_signature,) + outptut_signature[1:]

    def get_config(self):
        config = super().get_config()
        config.update({
            'bone_arch': self.bone_arch,
            'bone_init': self.bone_init,
            'pool_scales': self.pool_scales,
        })

        return config


def build_fba_matting(bone_arch='bit_m_r50x1_stride_8', bone_init='imagenet', psp_sizes=(1, 2, 3, 6)):
    inputs = [
        layers.Input(name='image', shape=[None, None, 3], dtype='uint8'),
        layers.Input(name='trimap', shape=[None, None, 1], dtype='uint8'),
    ]
    outputs = FBAMatting(bone_arch, bone_init, psp_sizes)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='fba_matting')

    return model

import tensorflow as tf
from tensorflow.keras import Model, layers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .decoder import Decoder
from .distance import Distance
from .fusion import Fusion
from .resnet import ResNet50
from .twomap import Twomap
from ...backbone.port.big_transfer import preprocess_input


@utils.register_keras_serializable(package='SegMe>FBAMatting')
class FBAMatting(layers.Layer):
    def __init__(self, pool_scales, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # image
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8'),  # trimap
        ]
        self.pool_scales = pool_scales

    def build(self, input_shape):
        self.twomap = Twomap()
        self.distance = Distance()
        self.encoder = ResNet50()
        self.decoder = Decoder(self.pool_scales)
        self.fusion = Fusion()

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        image, trimap = inputs  # TODO: scale to multiple of 8?
        image = tf.cast(image, self.compute_dtype)

        imnorm = preprocess_input(image)
        twomap = self.twomap(trimap)
        distance = self.distance(twomap)

        # Rescale twomap and distance to match preprocessed image
        featraw = layers.concatenate([
            imnorm,
            twomap * 2 - 1,  # Same scale as  imnorm
            distance * 2 - 1  # Same scale as  imnorm
        ], axis=-1)
        feats2, feats4, feats32 = self.encoder(featraw)

        imscal = image / 255.  # Same scale as twomap, alpha, foreground and background
        alfgbg = self.decoder([feats2, feats4, feats32, imscal, twomap])

        alpha, foreground, background = self.fusion([imscal, alfgbg])  # TODO: try twice?
        alpha = tf.round(alpha * 255.)
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
        config.update({'pool_scales': self.pool_scales})

        return config


def build_fba_matting(psp_sizes=(1, 2, 3, 6)):
    inputs = [
        layers.Input(name='image', shape=[None, None, 3], dtype='uint8'),
        layers.Input(name='trimap', shape=[None, None, 1], dtype='uint8'),
    ]
    outputs = FBAMatting(psp_sizes)(inputs)
    model = Model(inputs=inputs, outputs=outputs, name='fba_matting')

    return model
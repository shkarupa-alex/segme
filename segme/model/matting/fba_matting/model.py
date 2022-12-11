import tensorflow as tf
from keras import layers, models
from keras.applications.imagenet_utils import preprocess_input
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.model.matting.fba_matting.decoder import Decoder
from segme.model.matting.fba_matting.fusion import Fusion
from segme.model.matting.fba_matting.encoder import Encoder


@register_keras_serializable(package='SegMe>Model>Matting>FBAMatting')
class FBAMatting(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # image
            layers.InputSpec(ndim=4, axes={-1: 2}, dtype='uint8'),  # twomap
            layers.InputSpec(ndim=4, axes={-1: 6}, dtype='uint8'),  # distance
        ]

    def build(self, input_shape):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fusion = Fusion(dtype='float32')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        image, twomap, distance = inputs

        # Rescale twomap and distance to match preprocessed image
        featraw = tf.concat([image, twomap, distance], axis=-1)
        feats2, feats4, feats32 = self.encoder(featraw)

        imft32 = tf.cast(image, 'float32')
        imscal = imft32 / 255.
        imnorm = preprocess_input(imft32, mode='torch')
        alfgbg = self.decoder([
            feats2, feats4, feats32,
            imscal,  # scaled image
            imnorm,  # normalized image
            tf.cast(twomap, 'float32') / 255.  # scaled twomap
        ])

        alpha, foreground, background = self.fusion([imscal, alfgbg])

        return alfgbg, alpha, foreground, background

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        base_shape = input_shape[0][:-1]

        return base_shape + (7,), base_shape + (1,), base_shape + (3,), base_shape + (3,)

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return [tf.TensorSpec(dtype='float32', shape=os.shape) for os in outptut_signature]


def build_fba_matting():
    inputs = [
        layers.Input(name='image', shape=[None, None, 3], dtype='uint8'),
        layers.Input(name='twomap', shape=[None, None, 2], dtype='uint8'),
        layers.Input(name='distance', shape=[None, None, 6], dtype='uint8'),
    ]
    outputs = FBAMatting()(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='fba_matting')

    return model

import tensorflow as tf
from keras import backend, layers, models
from keras.utils.control_flow_util import smart_cond
from keras.utils.conv_utils import normalize_tuple
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.model.matting.matte_former.decoder import Decoder
from segme.model.matting.matte_former.encoder import Encoder
from segme.utils.matting.tf import alpha_trimap


@register_keras_serializable(package='SegMe>Model>Matting>MatteFormer')
class MatteFormer(layers.Layer):
    def __init__(self, filters, depths, radius, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # image
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8')  # trimap
        ]
        self.filters = filters
        self.depths = depths
        self.radius = normalize_tuple(radius, 2, 'radius')

    def build(self, input_shape):
        self.encoder = Encoder()
        self.decoder = Decoder(self.filters, self.depths)

        super().build(input_shape)

    def merge(self, coarse, fine, radius, training):
        radius = smart_cond(
            training,
            lambda: (1, radius),
            lambda: (radius // 2, radius // 2))

        alpha = tf.round(coarse * 255.)
        alpha = tf.cast(alpha, 'uint8')
        trimap = alpha_trimap(alpha, radius)
        weight = tf.cast(trimap == 128, 'float32')
        weight = tf.stop_gradient(weight)

        refine = fine * weight + coarse * (1. - weight)

        return refine

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = backend.learning_phase()

        features = self.encoder(inputs)
        alpha1, alpha4, alpha8 = self.decoder(features)

        refine4 = self.merge(alpha8, alpha4, self.radius[0], training)
        refine1 = self.merge(refine4, alpha1, self.radius[1], training)

        return refine1, refine1, refine4, alpha8

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = self.encoder.compute_output_shape(input_shape)
        output_shape = self.decoder.compute_output_shape(output_shape)

        return output_shape[:1] + output_shape

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return [tf.TensorSpec(dtype='float32', shape=os.shape) for os in outptut_signature]

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'depths': self.depths,
            'radius': self.radius
        })

        return config


def build_matte_former(filters=(256, 128, 64, 32), depths=(2, 3, 3, 2), radius=(30, 15)):
    inputs = [
        layers.Input(name='image', shape=[None, None, 3], dtype='uint8'),
        layers.Input(name='trimap', shape=[None, None, 1], dtype='uint8')
    ]
    outputs = MatteFormer(filters, depths, radius)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='matte_former')

    return model

import tensorflow as tf
from keras import layers, models
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .decoder import Decoder
from .encoder import Encoder


@register_keras_serializable(package='SegMe>MatteFormer')
class MatteFormer(layers.Layer):
    def __init__(self, filters, depths, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # image
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8')  # trimap
        ]
        self.filters = filters
        self.depths = depths

    def build(self, input_shape):
        self.encoder = Encoder()
        self.decoder = Decoder(self.filters, self.depths)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        features = self.encoder(inputs)
        outputs = self.decoder(features)

        return outputs[:1] + outputs

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
            'depths': self.depths
        })

        return config


def build_matte_former(filters=(256, 128, 64, 32), depths=(2, 3, 3, 2)):
    inputs = [
        layers.Input(name='image', shape=[None, None, 3], dtype='uint8'),
        layers.Input(name='trimap', shape=[None, None, 1], dtype='uint8')
    ]
    outputs = MatteFormer(filters, depths)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='matte_former')

    return model

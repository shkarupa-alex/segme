import tensorflow as tf
from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import ConvAct, Conv, Norm
from segme.common.sequent import Sequential
from segme.model.hrrn.encoder import Encoder
from segme.model.hrrn.decoder import Decoder


@register_keras_serializable(package='SegMe>Model>HRRN')
class HRRN(layers.Layer):
    """ Proposed in https://arxiv.org/abs/2108.03551 """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # image
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8')  # trimap
        ]

    @shape_type_conversion
    def build(self, input_shape):
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.encoder.build(input_shape)

        short_channels = self.encoder.compute_output_shape(input_shape[0][:-1] + (6,))
        short_channels = [shape[-1] for shape in short_channels[:-1]]
        short_channels = [max(channel, 32) for channel in short_channels]

        self.shorts = [
            Sequential([
                ConvAct(filters, 3, use_bias=False), Norm(),
                ConvAct(filters, 3, use_bias=False), Norm()])
            for filters in short_channels
        ]

        self.proj = Conv(2, 3)

        # import tensorflow_probability as tfp
        # self.dist = tfp.layers.DistributionLambda(
        #     lambda t: tfp.distributions.Normal(loc=t[..., :1], scale=t[..., 1:]), dtype='float32')
        # self.std = layers.Lambda(lambda rv_y: rv_y.stddev())

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        images, trimaps = inputs
        trimaps = tf.one_hot(trimaps[..., 0] // 86, 3, dtype='uint8') * 255

        combos = tf.concat([images, trimaps], axis=-1)
        combos = tf.cast(combos, self.compute_dtype)

        features = self.encoder(combos)
        shortcuts = [short(feat) for short, feat in zip(self.shorts, features[:-1])]
        outputs = self.decoder(shortcuts + features[-1:])

        outputs = self.proj(outputs)
        outputs = tf.cast(outputs, 'float32')

        mean, var = tf.split(outputs, 2, axis=-1)
        mean = tf.nn.tanh(mean) * 0.5 + 0.5
        var = tf.nn.sigmoid(var)
        mean_var = tf.concat([mean, var], axis=-1)

        # var = tf.nn.softplus(var) + 1e-3
        # # TODO: var = tf.nn.softplus(var * 0.05) + 1e-3
        #
        # outputs = tf.concat([mean, var], axis=-1)
        # outputs = self.dist(outputs)
        #
        # std = self.std(outputs)

        return mean, mean_var

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][:-1] + (1,)

        return output_shape, output_shape[:-1] + (2,)

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return [tf.TensorSpec(dtype='float32', shape=os.shape) for os in outptut_signature]


def build_hrrn():
    inputs = [
        layers.Input(name='image', shape=[None, None, 3], dtype='uint8'),
        layers.Input(name='trimap', shape=[None, None, 1], dtype='uint8')]
    outputs = HRRN()(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='hrrn')

    return model

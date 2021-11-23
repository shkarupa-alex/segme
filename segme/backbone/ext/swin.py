import tensorflow as tf
import tfswin
from functools import partial
from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFSwin')
class SwinReshape(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.length, self.channels = input_shape[1:]
        if None in {self.length, self.channels}:
            raise ValueError('Length and channel dimensions of the inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={1: self.length, 2: self.channels})

        # noinspection PyAttributeOutsideInit
        self.size = int(self.length ** 0.5)
        if self.size ** 2 != self.length:
            raise ValueError('Height and width of the inputs should be equal.')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = tf.reshape(inputs, [-1, self.size, self.size, self.channels])

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:1] + (self.size, self.size, self.channels)


def wrap_bone(model, size, channels, feats, init):
    input_shape = (size, size, channels)
    input_image = layers.Input(name='image', shape=input_shape, dtype=tf.uint8)
    input_prep = layers.Lambda(tfswin.preprocess_input, name='preprocess')(input_image)

    base_model = model(input_tensor=input_prep, include_top=False, weights=init)

    end_points = [ep for ep in feats if ep is not None]
    out_layers = [base_model.get_layer(name=name).output for name in end_points]
    out_layers = [SwinReshape()(ol) for ol in out_layers]

    down_stack = models.Model(inputs=input_image, outputs=out_layers)

    return down_stack


SwinTransformerTiny224 = partial(
    wrap_bone,
    tfswin.SwinTransformerTiny224,
    224
)

SwinTransformerSmall224 = partial(
    wrap_bone,
    tfswin.SwinTransformerSmall224,
    224
)

SwinTransformerBase224 = partial(
    wrap_bone,
    tfswin.SwinTransformerBase224,
    224
)

SwinTransformerBase384 = partial(
    wrap_bone,
    tfswin.SwinTransformerBase384,
    384
)

SwinTransformerLarge224 = partial(
    wrap_bone,
    tfswin.SwinTransformerLarge224,
    224
)

SwinTransformerLarge384 = partial(
    wrap_bone,
    tfswin.SwinTransformerLarge384,
    384
)

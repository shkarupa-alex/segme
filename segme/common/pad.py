import tensorflow as tf
from keras import backend, layers
from keras.saving import register_keras_serializable
from keras.src.utils.control_flow_util import smart_cond
from keras.src.utils.conv_utils import normalize_tuple
from segme.common.shape import get_shape


@register_keras_serializable(package='SegMe>Common')
class SymmetricPadding(layers.ZeroPadding2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if max(self.padding[0] + self.padding[1]) > 1:
            raise ValueError('Symmetric padding can lead to misbehavior when padding size > 1')

    def call(self, inputs):
        assert len(self.padding) == 2
        assert len(self.padding[0]) == 2
        assert len(self.padding[1]) == 2

        data_format = backend.image_data_format() if self.data_format is None else self.data_format
        assert 'channels_last' == data_format

        pattern = [[0, 0], list(self.padding[0]), list(self.padding[1]), [0, 0]]

        return tf.pad(inputs, pattern, mode='SYMMETRIC')


def with_divisible_pad(op, inputs, dividers, mode='CONSTANT', constant_values=0, dtype=None, name=None):
    with tf.name_scope(name or 'with_divisible_pad'):
        inputs = tf.convert_to_tensor(inputs, dtype)
        if 4 != inputs.shape.rank:
            raise ValueError('Expecting `inputs` rank to be 4.')

        dividers = normalize_tuple(dividers, 2, 'normalize_tuple')
        if 1 == max(dividers):
            raise ValueError('Nothing to pad: both multipliers equals to 1.')

        (inputs_height, inputs_width, inputs_channel), static_size = get_shape(inputs, axis=[1, 2, 3])
        h_pad = (dividers[0] - inputs_height % dividers[0]) % dividers[0]
        w_pad = (dividers[1] - inputs_width % dividers[1]) % dividers[1]
        with_pad = h_pad + w_pad > 0

        hb_pad, wb_pad = h_pad // 2, w_pad // 2
        ha_pad, wa_pad = h_pad - hb_pad, w_pad - wb_pad
        paddings = [[0, 0], [hb_pad, ha_pad], [wb_pad, wa_pad], [0, 0]]

        outputs = smart_cond(
            with_pad,
            lambda: tf.pad(inputs, paddings, mode=mode, constant_values=constant_values),
            lambda: tf.identity(inputs))

        if static_size:
            padded_shape = (inputs.shape[0], inputs_height + h_pad, inputs_width + w_pad, inputs_channel)
        else:
            padded_shape = (inputs.shape[0], None, None, inputs_channel)
        outputs.set_shape(padded_shape)

        pad_size = inputs_height + h_pad, inputs_width + w_pad
        pad_val = (hb_pad, ha_pad, wb_pad, wa_pad)
        outputs = op(outputs, pad_size=pad_size, pad_val=pad_val)
        (outputs_channel,), _ = get_shape(outputs, axis=[3])

        outputs = smart_cond(
            with_pad,
            lambda: tf.slice(outputs, [0, hb_pad, wb_pad, 0], [-1, inputs_height, inputs_width, outputs_channel]),
            lambda: tf.identity(outputs))
        outputs.set_shape(inputs.shape[:-1] + outputs.shape[-1:])

        return outputs

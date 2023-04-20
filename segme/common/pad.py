import tensorflow as tf
from keras import backend, layers
from keras.saving import register_keras_serializable
from keras.src.utils.control_flow_util import smart_cond
from keras.src.utils.conv_utils import normalize_tuple


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

        inputs_batch, inputs_height, inputs_width, inputs_channel = inputs.shape
        static_size = inputs_height is not None and inputs_width is not None
        if not static_size:
            inputs_batch, inputs_height, inputs_width = tf.unstack(tf.shape(inputs)[:3])
        else:
            inputs_batch = tf.shape(inputs)[0]

        h_pad = (dividers[0] - inputs_height % dividers[0]) % dividers[0]
        w_pad = (dividers[1] - inputs_width % dividers[1]) % dividers[1]
        hb_pad, wb_pad = h_pad // 2, w_pad // 2
        ha_pad, wa_pad = h_pad - hb_pad, w_pad - wb_pad

        paddings = [[0, 0], [hb_pad, ha_pad], [wb_pad, wa_pad], [0, 0]]
        with_pad = h_pad + w_pad > 0

        outputs = smart_cond(
            with_pad,
            lambda: tf.pad(inputs, paddings, mode=mode, constant_values=constant_values),
            lambda: tf.identity(inputs))
        padded_shape = (inputs.shape[0], None, None, inputs_channel)
        if static_size:
            padded_shape = (inputs.shape[0], inputs_height + h_pad, inputs_width + w_pad, inputs_channel)
        outputs.set_shape(padded_shape)

        pad_size = (inputs_batch, inputs_height + h_pad, inputs_width + w_pad)
        pad_val = (hb_pad, ha_pad, wb_pad, wa_pad)
        outputs = op(outputs, pad_size=pad_size, pad_val=pad_val)

        outputs_batch, outputs_height, outputs_width = tf.unstack(tf.shape(outputs)[:3])
        outputs_channel = outputs.shape[-1]

        assert_batch = tf.debugging.assert_equal(outputs_batch, pad_size[0])
        assert_height = tf.debugging.assert_equal(outputs_height, pad_size[1])
        assert_width = tf.debugging.assert_equal(outputs_width, pad_size[2])
        with tf.control_dependencies([assert_batch, assert_height, assert_width]):
            outputs = tf.identity(outputs)

        outputs = smart_cond(
            with_pad,
            lambda: tf.slice(outputs, [0, hb_pad, wb_pad, 0], [-1, inputs_height, inputs_width, -1]),
            lambda: tf.identity(outputs))
        outputs.set_shape(inputs.shape[:-1] + (outputs_channel,))

        return outputs

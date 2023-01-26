import tensorflow as tf
from keras import backend, layers
from keras.utils.conv_utils import normalize_tuple
from keras.saving.object_registration import register_keras_serializable


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

        inputs_shape = tf.unstack(tf.shape(inputs))
        inputs_batch, inputs_height, inputs_width, _ = inputs_shape
        inputs_height_, inputs_width_ = inputs.shape[1:3]

        h_pad = (dividers[0] - inputs_height % dividers[0]) % dividers[0]
        w_pad = (dividers[1] - inputs_width % dividers[1]) % dividers[1]

        h_pad_ = None if inputs_height_ is None else (dividers[0] - inputs_height_ % dividers[0]) % dividers[0]
        w_pad_ = None if inputs_width_ is None else (dividers[1] - inputs_width_ % dividers[1]) % dividers[1]

        hb_pad, wb_pad = h_pad // 2, w_pad // 2
        ha_pad, wa_pad = h_pad - hb_pad, w_pad - wb_pad
        paddings = [[0, 0], [hb_pad, ha_pad], [wb_pad, wa_pad], [0, 0]]

        padded_shape_ = (
            inputs.shape[0],
            None if inputs_height_ is None else inputs_height_ + h_pad_,
            None if inputs_width_ is None else inputs_width_ + w_pad_,
            inputs.shape[3])

        outputs = tf.pad(inputs, paddings, mode=mode, constant_values=constant_values)
        outputs.set_shape(padded_shape_)

        pad_size = (inputs_batch, inputs_height + h_pad, inputs_width + w_pad)
        outputs = op(outputs, pad_size=pad_size, pad_val=(hb_pad, ha_pad, wb_pad, wa_pad))

        outputs_shape = tf.unstack(tf.shape(outputs))
        outputs_batch, outputs_height, outputs_width, _ = outputs_shape

        assert_batch = tf.debugging.assert_equal(outputs_batch, inputs_batch)
        assert_height = tf.debugging.assert_equal(outputs_height, inputs_height + h_pad)
        assert_width = tf.debugging.assert_equal(outputs_width, inputs_width + w_pad)
        with tf.control_dependencies([assert_batch, assert_height, assert_width]):
            outputs = tf.identity(outputs)

        outputs = outputs[:, hb_pad:inputs_height + hb_pad, wb_pad: inputs_width + wb_pad]
        outputs.set_shape(inputs.shape[:-1] + outputs.shape[-1:])

        return outputs

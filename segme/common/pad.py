import tensorflow as tf
from keras import backend, layers
from keras.utils.control_flow_util import smart_cond
from keras.utils.conv_utils import normalize_data_format, normalize_tuple
from keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable(package='SegMe>Common')
class SymmetricPadding(layers.ZeroPadding2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if max(self.padding[0] + self.padding[1]) > 1:
            raise ValueError('Symmetric padding can lead to misbehavior when padding size > 1')

    def call(self, inputs):
        padding = self.padding
        data_format = self.data_format

        assert len(padding) == 2
        assert len(padding[0]) == 2
        assert len(padding[1]) == 2

        if data_format is None:
            data_format = backend.image_data_format()
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format: ' + str(data_format))

        if data_format == 'channels_first':
            pattern = [[0, 0], [0, 0], list(padding[0]), list(padding[1])]
        else:
            pattern = [[0, 0], list(padding[0]), list(padding[1]), [0, 0]]

        return tf.pad(inputs, pattern, mode='SYMMETRIC')


def with_divisible_pad(op, inputs, dividers, mode='CONSTANT', data_format=None, dtype=None, name=None):
    with tf.name_scope(name or 'with_divisible_pad'):
        inputs = tf.convert_to_tensor(inputs, dtype)
        if 4 != inputs.shape.rank:
            raise ValueError('Expecting `inputs` rank to be 4.')

        dividers = normalize_tuple(dividers, 2, 'normalize_tuple')
        if 1 == max(dividers):
            raise ValueError('Nothing to pad: both multipliers equals to 1.')

        data_format = normalize_data_format(data_format)

        inputs_shape = tf.unstack(tf.shape(inputs))
        if 'channels_last' == data_format:
            inputs_batch, inputs_height, inputs_width, _ = inputs_shape
            inputs_height_, inputs_width_ = inputs.shape[1:3]
        else:
            inputs_batch, _, inputs_height, inputs_width = inputs_shape
            inputs_height_, inputs_width_ = inputs.shape[2:4]

        h_pad = (dividers[0] - inputs_height % dividers[0]) % dividers[0]
        w_pad = (dividers[1] - inputs_width % dividers[1]) % dividers[1]

        h_pad_ = None if inputs_height_ is None else (dividers[0] - inputs_height_ % dividers[0]) % dividers[0]
        w_pad_ = None if inputs_width_ is None else (dividers[1] - inputs_width_ % dividers[1]) % dividers[1]

        if 'channels_last' == data_format:
            paddings = [[0, 0], [0, h_pad], [0, w_pad], [0, 0]]
            padded_shape_ = (
                inputs.shape[0],
                None if inputs_height_ is None else inputs_height_ + h_pad_,
                None if inputs_width_ is None else inputs_width_ + w_pad_,
                inputs.shape[3])
        else:
            paddings = [[0, 0], [0, 0], [0, h_pad], [0, w_pad]]
            padded_shape_ = (
                inputs.shape[0], inputs.shape[1],
                None if inputs_height_ is None else inputs_height_ + h_pad_,
                None if inputs_width_ is None else inputs_width_ + w_pad_)

        with_pad = h_pad + w_pad > 0
        outputs = smart_cond(
            with_pad,
            lambda: tf.pad(inputs, paddings, mode),
            lambda: tf.identity(inputs))
        outputs.set_shape(padded_shape_)

        outputs = op(outputs)

        outputs_shape = tf.unstack(tf.shape(outputs))
        if 'channels_last' == data_format:
            outputs_batch, outputs_height, outputs_width, _ = outputs_shape
        else:
            outputs_batch, _, outputs_height, outputs_width = outputs_shape

        assert_batch = tf.debugging.assert_equal(outputs_batch, inputs_batch)
        assert_height = tf.debugging.assert_equal(outputs_height, inputs_height + h_pad)
        assert_width = tf.debugging.assert_equal(outputs_width, inputs_width + w_pad)
        with tf.control_dependencies([assert_batch, assert_height, assert_width]):
            outputs = tf.identity(outputs)

        if 'channels_last' == data_format:
            outputs = smart_cond(
                with_pad,
                lambda: outputs[:, :inputs_height, :inputs_width],
                lambda: tf.identity(outputs))
        else:
            outputs = smart_cond(
                with_pad,
                lambda: outputs[:, :, :inputs_height, :inputs_width],
                lambda: tf.identity(outputs))
        outputs.set_shape(inputs.shape[:-1] + outputs.shape[-1:])

        return outputs

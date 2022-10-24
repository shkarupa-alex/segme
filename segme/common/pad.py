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


def with_multiple_pad(inputs, multipliers, mode='CONSTANT', data_format=None):
    inputs = tf.convert_to_tensor(inputs)
    if 4 != inputs.shape.rank:
        raise ValueError('Expecting `inputs` rank to be 4.')

    multipliers = normalize_tuple(multipliers, 2, 'normalize_tuple')
    if 1 == max(multipliers):
        raise ValueError('Nothing to pad: both multipliers equals to 1.')

    data_format = normalize_data_format(data_format)

    shape = tf.unstack(tf.shape(inputs))
    if 'channels_last' == data_format:
        batch, height, width, _ = shape
        height_, width_ = inputs.shape[1:3]
    else:
        batch, _, height, width = shape
        height_, width_ = inputs.shape[2:4]

    h_pad = (multipliers[0] - height % multipliers[0]) % multipliers[0]
    w_pad = (multipliers[1] - width % multipliers[1]) % multipliers[1]

    h_pad_ = None if height_ is None else (multipliers[0] - height_ % multipliers[0]) % multipliers[0]
    w_pad_ = None if width_ is None else (multipliers[1] - width_ % multipliers[1]) % multipliers[1]

    if 'channels_last' == data_format:
        paddings = [[0, 0], [0, h_pad], [0, w_pad], [0, 0]]
        shape_ = (
            inputs.shape[0],
            None if height_ is None else height_ + h_pad_,
            None if width_ is None else width_ + w_pad_,
            inputs.shape[3])
    else:
        paddings = [[0, 0], [0, 0], [0, h_pad], [0, w_pad]]
        shape_ = (
            inputs.shape[0], inputs.shape[1],
            None if height_ is None else height_ + h_pad_,
            None if width_ is None else width_ + w_pad_)

    with_pad = h_pad + w_pad > 0
    outputs = smart_cond(
        with_pad,
        lambda: tf.pad(inputs, paddings, mode),
        lambda: tf.identity(inputs))
    outputs.set_shape(shape_)

    def _unpad(padded):
        padded_shape = tf.unstack(tf.shape(padded))
        if 'channels_last' == data_format:
            padded_batch, padded_height, padded_width, _ = padded_shape
        else:
            padded_batch, _, padded_height, padded_width = padded_shape

        assert_batch = tf.debugging.assert_equal(padded_batch, batch)
        assert_height = tf.debugging.assert_equal(padded_height, height + h_pad)
        assert_width = tf.debugging.assert_equal(padded_width, width + w_pad)
        with tf.control_dependencies([assert_batch, assert_height, assert_width]):
            padded = tf.identity(padded)

        if 'channels_last' == data_format:
            unpadded = smart_cond(
                with_pad,
                lambda: padded[:, :height, :width],
                lambda: tf.identity(padded))
        else:
            unpadded = smart_cond(
                with_pad,
                lambda: padded[:, :, :height, :width],
                lambda: tf.identity(padded))
        unpadded.set_shape(inputs.shape[:-1] + (padded.shape[-1],))

        return unpadded

    return outputs, _unpad

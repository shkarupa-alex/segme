import tensorflow as tf
from segme.common.pad import with_divisible_pad

_PARTITION_TYPES = {'window_size', 'window_count', 'grid_size', 'grid_count'}


def partition_apply(inputs, height, width, part_type, size_count, dilation_rate=1, dtype=None, name=None):
    with tf.name_scope(name or 'partition_apply'):
        inputs = tf.convert_to_tensor(inputs, dtype)

        if 4 != inputs.shape.rank:
            raise ValueError('Expecting inputs rank to be 4.')

        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        if part_type not in _PARTITION_TYPES:
            raise ValueError('Unknown partition type.')

        height_blocks = height // (size_count * dilation_rate)
        width_blocks = width // (size_count * dilation_rate)

        if part_type in {'window_size', 'grid_count'}:
            outputs = tf.reshape(inputs, [
                -1, height_blocks, size_count, dilation_rate, width_blocks, size_count, dilation_rate, channels])
        else:
            outputs = tf.reshape(inputs, [
                -1, size_count, height_blocks, dilation_rate, size_count, width_blocks, dilation_rate, channels])

        if part_type in {'window_size', 'window_count'}:
            outputs = tf.transpose(outputs, [0, 1, 3, 4, 6, 2, 5, 7])
        else:
            outputs = tf.transpose(outputs, [0, 2, 3, 5, 6, 1, 4, 7])

        if part_type in {'window_size', 'grid_size'}:
            outputs = tf.reshape(outputs, [-1, (size_count ** 2), channels])
        else:
            outputs = tf.reshape(outputs, [-1, height_blocks * width_blocks, channels])

        return outputs


def partition_reverse(inputs, height, width, part_type, size_count, dilation_rate=1, dtype=None, name=None):
    with tf.name_scope(name or 'partition_reverse'):
        inputs = tf.convert_to_tensor(inputs, dtype)

        if 3 != inputs.shape.rank:
            raise ValueError('Expecting inputs rank to be 3.')

        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        if part_type not in _PARTITION_TYPES:
            raise ValueError('Unknown partition type.')

        height_blocks = height // (size_count * dilation_rate)
        width_blocks = width // (size_count * dilation_rate)

        if part_type in {'window_size', 'grid_size'}:
            outputs = tf.reshape(inputs, [
                -1, height_blocks, dilation_rate, width_blocks, dilation_rate, size_count, size_count, channels])
        else:
            outputs = tf.reshape(inputs, [
                -1, size_count, dilation_rate, size_count, dilation_rate, height_blocks, width_blocks, channels])

        if part_type in {'window_size', 'window_count'}:
            outputs = tf.transpose(outputs, [0, 1, 5, 2, 3, 6, 4, 7])
        else:
            outputs = tf.transpose(outputs, [0, 5, 1, 2, 6, 3, 4, 7])

        outputs = tf.reshape(outputs, [
            -1, height_blocks * size_count * dilation_rate, width_blocks * size_count * dilation_rate, channels])

        return outputs


def with_partition(op, inputs, part_type, size_count, dilation_rate=1, dtype=None, name=None):
    with tf.name_scope(name or 'with_partition'):
        inputs = tf.convert_to_tensor(inputs, dtype)

        if 4 != inputs.shape.rank:
            raise ValueError('Expecting inputs rank to be 4.')

        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        if part_type not in _PARTITION_TYPES:
            raise ValueError('Unknown partition type.')

        def _op(padded, with_pad, pad_size, pad_val):
            _, height, width = pad_size

            parted = partition_apply(padded, height, width, part_type, size_count, dilation_rate)
            parted = op(parted, with_pad=with_pad, pad_size=pad_size, pad_val=pad_val)
            parted = partition_reverse(parted, height, width, part_type, size_count, dilation_rate)

            return parted

        return with_divisible_pad(_op, inputs, size_count * dilation_rate)

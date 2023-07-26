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
            num_windows = height_blocks * width_blocks * tf.square(dilation_rate)
            outputs = tf.reshape(outputs, [-1, num_windows, (size_count ** 2), channels])
        else:
            num_windows = tf.square(size_count * dilation_rate)
            outputs = tf.reshape(outputs, [-1, num_windows, height_blocks * width_blocks, channels])

        return outputs


def partition_reverse(inputs, height, width, part_type, size_count, dilation_rate=1, dtype=None, name=None):
    with tf.name_scope(name or 'partition_reverse'):
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

        if part_type not in _PARTITION_TYPES:
            raise ValueError('Unknown partition type.')

        def _op(padded, pad_size, pad_val):
            height, width = pad_size

            parted = partition_apply(padded, height, width, part_type, size_count, dilation_rate, dtype)
            parted = op(parted, pad_size=pad_size, pad_val=pad_val)
            parted = partition_reverse(parted, height, width, part_type, size_count, dilation_rate, dtype)

            return parted

        return with_divisible_pad(_op, inputs, size_count * dilation_rate)


def halo_partition(inputs, height, width, window_size, halo_size, dilation_rate=1, dtype=None, name=None):
    with tf.name_scope(name or 'halo_partition'):
        inputs = tf.convert_to_tensor(inputs, dtype)

        if 4 != inputs.shape.rank:
            raise ValueError('Expecting inputs rank to be 4.')

        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        if halo_size < window_size:
            raise ValueError('Halo size must be not less then window size.')

        if (halo_size - window_size) % 2:
            raise ValueError('Halo size must be symmetric around window size.')

        halo_kernel = [1, halo_size * dilation_rate, halo_size * dilation_rate, 1]
        halo_stride = [1, window_size * dilation_rate, window_size * dilation_rate, 1]

        height_blocks = height // (window_size * dilation_rate)
        width_blocks = width // (window_size * dilation_rate)
        num_windows = height_blocks * width_blocks * tf.square(dilation_rate)

        outputs = tf.image.extract_patches(inputs, halo_kernel, halo_stride, [1] * 4, padding='SAME')

        # Non-fused implementation with window partition step
        # halo_factor = halo_size / window_size
        # if halo_factor % 1:
        #     halo_height = tf.cast(tf.math.ceil(tf.cast(height, 'float32') * halo_factor), height.dtype)
        #     halo_width = tf.cast(tf.math.ceil(tf.cast(width, 'float32') * halo_factor), width.dtype)
        # else:
        #     halo_height, halo_width = height * int(halo_factor), width * int(halo_factor)
        # outputs = tf.reshape(outputs, [
        #     -1, height_blocks, width_blocks, halo_size, dilation_rate, halo_size, dilation_rate, channels])
        # outputs = tf.transpose(outputs, [0, 1, 3, 4, 2, 5, 6, 7])
        # outputs = tf.reshape(outputs, [-1, halo_height, halo_width, channels])
        # outputs = partition_apply(outputs, halo_height, halo_width, 'window_size', halo_size, dilation_rate)

        outputs = tf.reshape(outputs, [
            -1, height_blocks, width_blocks, halo_size, dilation_rate, halo_size, dilation_rate, channels])
        outputs = tf.transpose(outputs, [0, 1, 4, 2, 6, 3, 5, 7])
        outputs = tf.reshape(outputs, [-1, num_windows, (halo_size ** 2), channels])

        return outputs


def partition_apply_fused(
        inputs, height, width, part_type, size_count, num_heads, dilation_rate=1, dtype=None, name=None):
    with tf.name_scope(name or 'partition_apply_fused'):
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
                -1, height_blocks, size_count, dilation_rate, width_blocks, size_count, dilation_rate,
                num_heads, channels // num_heads])
        else:
            outputs = tf.reshape(inputs, [
                -1, size_count, height_blocks, dilation_rate, size_count, width_blocks, dilation_rate,
                num_heads, channels // num_heads])

        if part_type in {'window_size', 'window_count'}:
            outputs = tf.transpose(outputs, [0, 1, 3, 4, 6, 7, 2, 5, 8])
        else:
            outputs = tf.transpose(outputs, [0, 2, 3, 5, 6, 7, 1, 4, 8])

        if part_type in {'window_size', 'grid_size'}:
            num_windows = height_blocks * width_blocks * tf.square(dilation_rate)
            outputs = tf.reshape(
                outputs, [-1, num_windows, num_heads, (size_count ** 2), channels // num_heads])
        else:
            num_windows = tf.square(size_count * dilation_rate)
            outputs = tf.reshape(
                outputs, [-1, num_windows, num_heads, height_blocks * width_blocks, channels // num_heads])

        return outputs


def partition_reverse_fused(
        inputs, height, width, part_type, size_count, num_heads, dilation_rate=1, dtype=None, name=None):
    with tf.name_scope(name or 'partition_reverse_fused'):
        inputs = tf.convert_to_tensor(inputs, dtype)

        if 5 != inputs.shape.rank:
            raise ValueError('Expecting inputs rank to be 5.')

        head_channels = inputs.shape[-1]
        if head_channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        if part_type not in _PARTITION_TYPES:
            raise ValueError('Unknown partition type.')

        height_blocks = height // (size_count * dilation_rate)
        width_blocks = width // (size_count * dilation_rate)
        full_channels = num_heads * head_channels

        if part_type in {'window_size', 'grid_size'}:
            outputs = tf.reshape(inputs, [
                -1, height_blocks, dilation_rate, width_blocks, dilation_rate, num_heads, size_count, size_count,
                head_channels])
        else:
            outputs = tf.reshape(inputs, [
                -1, size_count, dilation_rate, size_count, dilation_rate, num_heads, height_blocks, width_blocks,
                head_channels])

        if part_type in {'window_size', 'window_count'}:
            outputs = tf.transpose(outputs, [0, 1, 6, 2, 3, 7, 4, 8, 5])
        else:
            outputs = tf.transpose(outputs, [0, 6, 1, 2, 7, 3, 4, 8, 5])

        outputs = tf.reshape(outputs, [
            -1, height_blocks * size_count * dilation_rate, width_blocks * size_count * dilation_rate, full_channels])

        return outputs


def with_partition_fused(op, inputs, part_type, size_count, num_heads, dilation_rate=1, dtype=None, name=None):
    with tf.name_scope(name or 'with_partition_fused'):
        inputs = tf.convert_to_tensor(inputs, dtype)

        if 4 != inputs.shape.rank:
            raise ValueError('Expecting inputs rank to be 4.')

        if part_type not in _PARTITION_TYPES:
            raise ValueError('Unknown partition type.')

        def _op(padded, pad_size, pad_val):
            height, width = pad_size

            parted = partition_apply_fused(
                padded, height, width, part_type, size_count, num_heads, dilation_rate, dtype)
            parted = op(parted, pad_size=pad_size, pad_val=pad_val)
            parted = partition_reverse_fused(
                parted, height, width, part_type, size_count, num_heads, dilation_rate, dtype)

            return parted

        return with_divisible_pad(_op, inputs, size_count * dilation_rate)


def halo_partition_fused(
        inputs, height, width, window_size, halo_size, num_heads, dilation_rate=1, dtype=None, name=None):
    with tf.name_scope(name or 'halo_partition_fused'):
        inputs = tf.convert_to_tensor(inputs, dtype)

        if 4 != inputs.shape.rank:
            raise ValueError('Expecting inputs rank to be 4.')

        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        if halo_size < window_size:
            raise ValueError('Halo size must be not less then window size.')

        if (halo_size - window_size) % 2:
            raise ValueError('Halo size must be symmetric around window size.')

        halo_kernel = [1, halo_size * dilation_rate, halo_size * dilation_rate, 1]
        halo_stride = [1, window_size * dilation_rate, window_size * dilation_rate, 1]

        height_blocks = height // (window_size * dilation_rate)
        width_blocks = width // (window_size * dilation_rate)
        num_windows = height_blocks * width_blocks * tf.square(dilation_rate)

        outputs = tf.image.extract_patches(inputs, halo_kernel, halo_stride, [1] * 4, padding='SAME')

        outputs = tf.reshape(outputs, [
            -1, height_blocks, width_blocks, halo_size, dilation_rate, halo_size, dilation_rate, num_heads,
            channels // num_heads])
        outputs = tf.transpose(outputs, [0, 1, 4, 2, 6, 7, 3, 5, 8])
        outputs = tf.reshape(outputs, [-1, num_windows, num_heads, halo_size ** 2, channels // num_heads])

        return outputs

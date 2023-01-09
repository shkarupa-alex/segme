import tensorflow as tf
from tensorflow.python.framework.ops import gradient_registry
from tensorflow.python.ops.array_grad import _IndexedSlicesToTensorNoWarning

del gradient_registry._registry['ExtractImagePatches']


@tf.RegisterGradient('ExtractImagePatches')
def _ExtractImagePatchesGrad(op, grad):
    input_shape = tf.shape(op.inputs[0], out_type='int64')
    batch, input_height, input_width, channels = tf.unstack(input_shape)

    _, kernel_height, kernel_width, _ = op.get_attr('ksizes')

    output_shape = tf.shape(op.outputs[0], out_type='int64')
    output_height, output_width = tf.unstack(output_shape[1:3])

    # Create indices matrix for input tensor. Note that 0 is preserved for padding location,
    # so indices for input start from 1 to 1 + input_height * input_width.
    input_indices = input_height * input_width + 1
    input_idx = tf.reshape(tf.range(1, input_indices, dtype='float32'), (1, input_height, input_width, 1))
    idx_patches = tf.image.extract_patches(
        input_idx, op.get_attr('ksizes'), op.get_attr('strides'), op.get_attr('rates'), op.get_attr('padding'))
    idx_patches = tf.cast(idx_patches, 'int64')

    grad_expanded = tf.transpose(
        tf.reshape(
            _IndexedSlicesToTensorNoWarning(grad),
            (batch, output_height, output_width, kernel_height, kernel_width, channels)),
        (1, 2, 3, 4, 0, 5))
    grad_flat = tf.reshape(grad_expanded, (-1, batch * channels))

    segment_ids = tf.reshape(idx_patches, [-1]) - 1

    grad_out = tf.math.unsorted_segment_sum(grad_flat, segment_ids, num_segments=input_indices - 1)
    grad_out = tf.reshape(grad_out, (input_height, input_width, batch, channels))
    grad_out = tf.transpose(grad_out, (2, 0, 1, 3))

    return [grad_out]


def extract_patches_xla(images, sizes, strides, rates, padding, name=None):
    return tf.image.extract_patches(images, sizes, strides, rates, padding, name)

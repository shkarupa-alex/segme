import tensorflow as tf


def make_coord(batch, height, width, dtype='float32'):
    # Make coordinates at grid centers.

    height_ = 1. / tf.cast(height, 'float32')
    width_ = 1. / tf.cast(width, 'float32')

    vertical = height_ - 1. + 2 * height_ * tf.range(height, dtype='float32')
    horizontal = width_ - 1. + 2 * width_ * tf.range(width, dtype='float32')

    mesh = tf.meshgrid(vertical, horizontal, indexing='ij')
    join = tf.cast(tf.stack(mesh, axis=-1), dtype)
    outputs = tf.tile(join[None], [batch, 1, 1, 1])

    return outputs

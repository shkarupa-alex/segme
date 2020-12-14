import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


@tf.keras.utils.register_keras_serializable(package='SegMe')
class PixelPositionAwareBinaryLoss(LossFunctionWrapper):
    """ Initially proposed in: 'F3Net: Fusion, Feedback and Focus for Salient Object Detection'

    Implements Equation [6] in https://arxiv.org/pdf/1911.11445.pdf
    """

    def __init__(
            self, from_logits=False, gamma=5, ksize=31, reduction=tf.keras.losses.Reduction.AUTO,
            name='pixel_position_aware_binary_loss'):
        super().__init__(
            pixel_position_aware_binary_loss, reduction=reduction, name=name, from_logits=from_logits, gamma=gamma,
            ksize=ksize)


@tf.keras.utils.register_keras_serializable(package='SegMe')
def pixel_position_aware_binary_loss(y_true, y_pred, from_logits=False, gamma=5, ksize=31):
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred.shape.assert_has_rank(4)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    weight = 1 + gamma * tf.abs(tf.nn.avg_pool2d(y_true, ksize=ksize, strides=1, padding='SAME') - y_true)

    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    wbce = tf.reduce_sum(weight * bce, axis=[1, 2]) / tf.reduce_sum(weight, axis=[1, 2])

    _y_pred = tf.nn.sigmoid(y_pred) if from_logits else y_pred
    intersection = tf.reduce_sum(_y_pred * y_true * weight, axis=[1, 2])
    union = tf.reduce_sum((_y_pred + y_true) * weight, axis=[1, 2])
    wiou = 1. - (intersection + 1.) / (union - intersection + 1.)

    print(wbce, wiou)

    return tf.reduce_mean(tf.concat([wbce, wiou], axis=-1), axis=-1)

import tensorflow as tf
from keras import losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from tensorflow_addons.image import euclidean_dist_transform


@register_keras_serializable(package='SegMe')
class BoundarySigmoidLoss(losses.LossFunctionWrapper):
    """ Proposed in: 'Boundary loss for highly unbalanced segmentation'

    Implements Equations from https://arxiv.org/pdf/1812.07032v4.pdf
    """

    def __init__(self, from_logits=False, reduction=Reduction.AUTO, name='boundary_sigmoid_loss'):
        super().__init__(boundary_sigmoid_loss, reduction=reduction, name=name, from_logits=from_logits)


def boundary_sigmoid_loss(y_true, y_pred, from_logits):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype='uint8')

    assert_true_rank = tf.assert_rank(y_true, 4)
    assert_pred_rank = tf.assert_rank(y_pred, 4)

    with tf.control_dependencies([assert_true_rank, assert_pred_rank]):
        if from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        y_false = 1 - y_true

        d_true = euclidean_dist_transform(tf.cast(y_true, dtype='uint8'), dtype=y_pred.dtype)
        d_false = euclidean_dist_transform(tf.cast(y_false, dtype='uint8'), dtype=y_pred.dtype)

        distance = d_false * tf.cast(y_false, dtype=y_pred.dtype) - (d_true - 1.) * tf.cast(y_true, dtype=y_pred.dtype)
        axis_hwc = list(range(1, y_pred.shape.ndims))
        has_true = tf.reduce_any(y_true == 1, axis=axis_hwc, keepdims=True)
        distance = tf.where(has_true, distance, 0.)
        distance = tf.stop_gradient(distance)

        loss = y_pred * distance

        return tf.reduce_mean(loss, axis=-1)

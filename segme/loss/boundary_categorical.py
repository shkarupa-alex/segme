import tensorflow as tf
from keras import losses
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from tensorflow_addons.image import euclidean_dist_transform


@register_keras_serializable(package='SegMe')
class BoundarySparseCategoricalLoss(losses.LossFunctionWrapper):
    """ Proposed in: 'Boundary loss for highly unbalanced segmentation'

    Implements Equation (5) from https://arxiv.org/pdf/1812.07032v4.pdf
    """

    def __init__(self, from_logits=False, reduction=Reduction.AUTO, name='boundary_sparse_categorical_loss'):
        super().__init__(boundary_sparse_categorical_loss, reduction=reduction, name=name, from_logits=from_logits)


def boundary_sparse_categorical_loss(y_true, y_pred, from_logits):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype='uint8')

    channels = y_pred.shape[-1]
    if channels is None:
        raise ValueError('Channel dimension of the predictions should be defined. Found `None`.')

    assert_true_rank = tf.assert_rank(y_true, 4)
    assert_pred_rank = tf.assert_rank(y_pred, 4)

    with tf.control_dependencies([assert_true_rank, assert_pred_rank]):
        if from_logits:
            if 1 == channels:
                y_pred = tf.nn.sigmoid(y_pred)
            else:
                y_pred = tf.nn.sigmoid(y_pred)

        axis_hwc = list(range(1, y_pred.shape.ndims))
        has_true = tf.reduce_any(y_true == 1, axis=axis_hwc, keepdims=True)
        has_false = tf.reduce_any(y_true == 0, axis=axis_hwc, keepdims=True)

        if 1 == channels:
            y_true = tf.one_hot(y_true[..., 0], 2, dtype='uint8')
            y_pred = tf.concat([1. - y_pred, y_pred], axis=-1)

        y_false = 1 - y_true

        d_true = euclidean_dist_transform(y_true, dtype=y_pred.dtype)
        d_false = euclidean_dist_transform(y_false, dtype=y_pred.dtype)

        distance = d_false * tf.cast(y_false, dtype=y_pred.dtype) - (d_true - 1.) * tf.cast(y_true, dtype=y_pred.dtype)
        distance = tf.where(has_true & has_false, distance, 0.)
        distance = tf.stop_gradient(distance)

        loss = y_pred * distance

        return tf.reduce_mean(loss, axis=-1)

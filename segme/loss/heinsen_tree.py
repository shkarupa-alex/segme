import numpy as np
import tensorflow as tf
from keras import losses
from keras.saving import register_keras_serializable
from keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, to_logits, weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe>Loss')
class HeinsenTreeLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Tree Methods for Hierarchical Classification in Parallel'

    Implements equation [1-13] in https://arxiv.org/pdf/2209.10288.pdf
    """

    def __init__(self, tree_paths, crossentropy='categorical', label_smoothing=0., from_logits=False,
                 reduction=Reduction.AUTO, name='heinsen_tree_loss'):
        super().__init__(heinsen_tree_loss, reduction=reduction, name=name, tree_paths=tree_paths,
                         crossentropy=crossentropy, label_smoothing=label_smoothing, from_logits=from_logits)


def heinsen_tree_loss(y_true, y_pred, sample_weight, tree_paths, crossentropy, label_smoothing, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype='int32', rank=None, channel='sparse')

    y_pred, from_logits = to_logits(y_pred, from_logits), True

    tree_classes = len(tree_paths)
    if y_pred.shape[-1] != tree_classes:
        raise ValueError(
            f'Number of classes in logits must match one in the tree. Got {y_pred.shape[-1]} vs {tree_classes}.')

    class_path = {p[-1]: p for p in tree_paths}
    if len(class_path) != tree_classes:
        raise ValueError('All tree paths must ends with unique class.')
    bad_classes = set(class_path.keys()) - set(range(tree_classes))
    if bad_classes:
        raise ValueError(f'Tree paths contain invalid classes: {bad_classes}.')

    num_levels = max(len(path) for path in tree_paths)
    tree_paths = [class_path[i] for i in range(tree_classes)]
    tree_paths = np.array([path + [-1] * (num_levels - len(path)) for path in tree_paths])
    valid_mask = (tree_paths.T == np.arange(tree_classes)[None])

    tree_paths = tf.cast(tree_paths, 'int32')
    valid_mask = tf.cast(valid_mask[None], 'bool')

    y_true_tree = tf.reshape(y_true, [-1])
    y_true_tree = tf.gather(tree_paths, y_true_tree)

    y_pred_tree = tf.reshape(y_pred, [-1, 1, tree_classes])
    y_pred_tree = tf.where(valid_mask, y_pred_tree, -100.)

    y_valid_tree = y_true_tree != -1
    y_true_tree = y_true_tree[y_valid_tree]
    y_pred_tree = y_pred_tree[y_valid_tree]

    if 'categorical' == crossentropy and 0. == label_smoothing:
        loss = losses.sparse_categorical_crossentropy(y_true_tree, y_pred_tree, from_logits=from_logits)
    elif 'categorical' == crossentropy:
        y_true_tree_1h = tf.one_hot(y_true_tree, tree_classes, dtype='float32')
        loss = losses.categorical_crossentropy(
            y_true_tree_1h, y_pred_tree, from_logits=from_logits, label_smoothing=label_smoothing)
    elif 'binary' == crossentropy:
        y_true_tree_1h = tf.one_hot(y_true_tree, tree_classes, dtype='float32')
        loss = losses.binary_crossentropy(
            y_true_tree_1h, y_pred_tree, from_logits=from_logits, label_smoothing=label_smoothing)
    else:
        raise ValueError(f'Unsupported cross entropy type: {crossentropy}')

    level_weight = tf.cast(y_valid_tree, loss.dtype) * tf.range(1, num_levels + 1, dtype=loss.dtype)[None]
    level_weight /= tf.reduce_sum(level_weight, axis=-1, keepdims=True)
    level_weight = level_weight[y_valid_tree]

    sample_segment = tf.cast(y_valid_tree, 'int32') * tf.range(0, tf.size(y_true))[:, None]
    sample_segment = tf.reshape(sample_segment[y_valid_tree], [-1])

    loss = tf.math.unsorted_segment_sum(loss * level_weight, sample_segment, num_segments=tf.size(y_true))
    loss = tf.reshape(loss, tf.shape(y_true))

    loss = weighted_loss(loss, sample_weight)

    return loss

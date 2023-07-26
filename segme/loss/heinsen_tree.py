import math
import numpy as np
import tensorflow as tf
from keras import losses
from keras.saving import register_keras_serializable
from keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.common_loss import validate_input, to_logits, weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper
from segme.common.shape import get_shape


@register_keras_serializable(package='SegMe>Loss')
class HeinsenTreeLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Tree Methods for Hierarchical Classification in Parallel'

    Implements equation [1-13] in https://arxiv.org/pdf/2209.10288.pdf
    """

    def __init__(self, tree_paths, cross_entropy='categorical', label_smoothing=0., level_weighting='mean',
                 from_logits=False, reduction=Reduction.AUTO, name='heinsen_tree_loss'):
        super().__init__(heinsen_tree_loss, reduction=reduction, name=name, tree_paths=tree_paths,
                         cross_entropy=cross_entropy, label_smoothing=label_smoothing, level_weighting=level_weighting,
                         from_logits=from_logits)


def heinsen_tree_loss(y_true, y_pred, sample_weight, tree_paths, cross_entropy, label_smoothing, level_weighting,
                      from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype='int32', rank=None, channel='sparse')

    y_pred, from_logits = to_logits(y_pred, from_logits), True

    tree_classes = len(tree_paths)
    if y_pred.shape[-1] != tree_classes:
        raise ValueError(
            f'Number of classes in logits must match one in the tree. Got {y_pred.shape[-1]} vs {tree_classes}.')

    if 0. == label_smoothing:
        invalid_logit = -100.
    elif 'categorical' == cross_entropy:
        label_smoothing = max(label_smoothing, 1e-6)
        invalid_logit = math.log(
            label_smoothing / (tree_classes - tree_classes * label_smoothing + label_smoothing))
    elif 'binary' == cross_entropy:
        label_smoothing = max(label_smoothing, 1e-6)
        invalid_logit = math.log(label_smoothing / (2 - label_smoothing))
    else:
        raise ValueError(f'Unsupported cross entropy type: {cross_entropy}')

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
    level_size = [np.unique(level).size - int(-1 in level) for level in tree_paths.T]
    level_size = 1. / np.array([level_size[i] for i in np.where(valid_mask)[0]])

    tree_paths = tf.convert_to_tensor(tree_paths, dtype='int32')
    valid_mask = tf.convert_to_tensor(valid_mask[None], dtype='bool')
    level_size = tf.convert_to_tensor(level_size, dtype=y_pred.dtype)

    y_true_tree = tf.reshape(y_true, [-1])
    y_true_tree = tf.gather(tree_paths, y_true_tree)

    y_pred_tree = tf.reshape(y_pred, [-1, 1, tree_classes])
    y_pred_tree = tf.where(valid_mask, y_pred_tree, invalid_logit)

    y_valid_tree = y_true_tree != -1
    y_true_tree = y_true_tree[y_valid_tree]
    y_pred_tree = y_pred_tree[y_valid_tree]

    if 'categorical' == cross_entropy and 0. == label_smoothing:
        loss = losses.sparse_categorical_crossentropy(y_true_tree, y_pred_tree, from_logits=from_logits)
    elif 'categorical' == cross_entropy:
        y_true_tree_1h = tf.one_hot(y_true_tree, tree_classes, dtype=y_pred.dtype)
        loss = losses.categorical_crossentropy(
            y_true_tree_1h, y_pred_tree, from_logits=from_logits, label_smoothing=label_smoothing)
    elif 'binary' == cross_entropy:
        y_true_tree_1h = tf.one_hot(y_true_tree, tree_classes, dtype=y_pred.dtype)
        loss = losses.binary_crossentropy(
            y_true_tree_1h[..., None], y_pred_tree[..., None], from_logits=from_logits, label_smoothing=label_smoothing)

        if label_smoothing > 0.:
            y_mask_tree = tf.ones_like(y_pred, dtype='bool')
            y_mask_tree = tf.reshape(y_mask_tree, [-1, 1, tree_classes])
            y_mask_tree = tf.where(valid_mask, y_mask_tree, False)
            y_mask_tree = y_mask_tree[y_valid_tree]
            loss = tf.where(y_mask_tree, loss, 0.)

        loss = tf.reduce_sum(loss, axis=-1) * tf.gather(level_size, y_true_tree)
    else:
        raise ValueError(f'Unsupported cross entropy type: {cross_entropy}')

    if 'mean' == level_weighting:
        level_weight = tf.cast(y_valid_tree, loss.dtype)
    elif 'linear' == level_weighting:
        level_range = tf.range(1, num_levels + 1, dtype=loss.dtype)
        level_weight = tf.cast(y_valid_tree, loss.dtype) * level_range[None]
    elif 'log' == level_weighting:
        level_range = tf.range(2, num_levels + 2, dtype=loss.dtype)
        level_range = tf.math.log(level_range)
        level_weight = tf.cast(y_valid_tree, loss.dtype) * level_range[None]
    elif 'pow' == level_weighting:
        level_range = tf.range(1, num_levels + 1, dtype=loss.dtype)
        level_range = (1. + 1. / num_levels) ** level_range
        level_weight = tf.cast(y_valid_tree, loss.dtype) * level_range[None]
    elif 'cumsum' == level_weighting:
        level_range = tf.range(1, num_levels + 1, dtype=loss.dtype)
        level_range = tf.cumsum(level_range)
        level_weight = tf.cast(y_valid_tree, loss.dtype) * level_range[None]
    else:
        raise ValueError(f'Unknown level weighting mode {level_weighting}')

    level_weight /= tf.reduce_sum(level_weight, axis=-1, keepdims=True)
    level_weight = level_weight[y_valid_tree]
    loss *= level_weight

    sample_segment = tf.cast(y_valid_tree, 'int32') * tf.range(tf.size(y_true))[:, None]
    sample_segment = tf.reshape(sample_segment[y_valid_tree], [-1])
    loss = tf.math.unsorted_segment_sum(loss, sample_segment, num_segments=tf.size(y_true))

    shape, _ = get_shape(y_true)
    loss = tf.reshape(loss, shape)

    loss = weighted_loss(loss, sample_weight)

    return loss

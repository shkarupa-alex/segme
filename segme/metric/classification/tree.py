import numpy as np
import tensorflow as tf
from keras.metrics import Metric
from keras.saving import register_keras_serializable


@register_keras_serializable(package='SegMe>Metric>Classification')
class HeinsenTreeAccuracy(Metric):
    def __init__(self, tree_paths, name='heinsen_tree_accuracy', dtype=None):
        """Creates a `HeinsenTreeAccuracy` instance for tree classification task.

        Args:
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name=name, dtype=dtype)

        self.tree_paths = tree_paths

        self.tree_classes = len(self.tree_paths)
        class_path = {p[-1]: p for p in self.tree_paths}
        if len(class_path) != self.tree_classes:
            raise ValueError('All tree paths must ends with unique class.')
        bad_classes = set(class_path.keys()) - set(range(self.tree_classes))
        if bad_classes:
            raise ValueError(f'Tree paths contain invalid classes: {bad_classes}.')

        self.num_levels = max(len(path) for path in self.tree_paths)
        paths_mask = [class_path[i] for i in range(self.tree_classes)]
        paths_mask = np.array([path + [-1] * (self.num_levels - len(path)) for path in paths_mask])
        valid_mask = paths_mask.T == np.arange(self.tree_classes)[None]
        self.paths_mask = tf.cast(paths_mask, 'int32')
        self.valid_mask = tf.cast(valid_mask[None], 'bool')

        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, self.dtype)
        if y_pred.shape[-1] != self.tree_classes:
            raise ValueError(
                f'Number of classes in logits must match one in the tree. '
                f'Got {y_pred.shape[-1]} vs {self.tree_classes}.')

        y_true_tree = tf.reshape(y_true, [-1])
        y_true_tree = tf.gather(self.paths_mask, y_true_tree)

        if sample_weight is not None:
            sample_weight = tf.reshape(sample_weight, [-1])

        y_pred_tree = tf.reshape(y_pred, [-1, 1, self.tree_classes])
        y_pred_tree = tf.where(self.valid_mask, y_pred_tree, y_pred.dtype.min)

        y_pred_tree = tf.argmax(y_pred_tree, axis=-1, output_type=y_true.dtype)
        y_match_tree = tf.cast(y_true_tree == y_pred_tree, self.dtype)
        y_valid_tree = tf.cast(y_true_tree != -1, self.dtype)

        y_match_tree = tf.split(y_match_tree, self.num_levels, axis=-1)
        c_match_tree = y_match_tree[:1]
        for i in range(1, self.num_levels):
            c_match_tree.append(tf.minimum(c_match_tree[i - 1], y_match_tree[i]))
        y_match_tree = tf.concat(c_match_tree, axis=-1)

        values = tf.reduce_sum(y_match_tree, axis=-1) / tf.reduce_sum(y_valid_tree, axis=-1)
        if sample_weight is not None:
            values *= sample_weight

        value_sum = tf.reduce_sum(values)
        with tf.control_dependencies([value_sum]):
            update_total_op = self.total.assign_add(value_sum)

        if sample_weight is None:
            num_values = tf.cast(tf.size(values), self.dtype)
        else:
            num_values = tf.reduce_sum(sample_weight)
        with tf.control_dependencies([update_total_op]):
            return self.count.assign_add(num_values)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def get_config(self):
        config = super().get_config()
        config.update({'tree_paths': self.tree_paths})

        return config

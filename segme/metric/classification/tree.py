import numpy as np
from keras.src import backend
from keras.src import ops
from keras.src.metrics import Metric
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Metric>Classification")
class HeinsenTreeAccuracy(Metric):
    def __init__(self, tree_paths, name="heinsen_tree_accuracy", dtype=None):
        """Creates a `HeinsenTreeAccuracy` instance for tree classification
        task.

        Args:
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name=name, dtype=dtype)

        self.tree_paths = tree_paths

        self.tree_classes = len(self.tree_paths)
        class_path = {p[-1]: p for p in self.tree_paths}
        if len(class_path) != self.tree_classes:
            raise ValueError("All tree paths must ends with unique class.")
        bad_classes = set(class_path.keys()) - set(range(self.tree_classes))
        if bad_classes:
            raise ValueError(
                f"Tree paths contain invalid classes: {bad_classes}."
            )

        self.num_levels = max(len(path) for path in self.tree_paths)
        paths_mask = [class_path[i] for i in range(self.tree_classes)]
        paths_mask = np.array(
            [path + [-1] * (self.num_levels - len(path)) for path in paths_mask]
        )
        valid_mask = paths_mask.T == np.arange(self.tree_classes)[None]
        self.paths_mask = ops.cast(paths_mask, "int32")
        self.valid_mask = ops.cast(valid_mask[None], "bool")

        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = backend.convert_to_tensor(y_pred, dtype=self.dtype)
        if y_pred.shape[-1] != self.tree_classes:
            raise ValueError(
                f"Number of classes in logits must match one in the tree. "
                f"Got {y_pred.shape[-1]} vs {self.tree_classes}."
            )

        y_true_tree = ops.reshape(y_true, [-1])
        y_true_tree = ops.take(self.paths_mask, y_true_tree, axis=0)

        if sample_weight is not None:
            sample_weight = ops.reshape(sample_weight, [-1])

        y_pred_tree = ops.reshape(y_pred, [-1, 1, self.tree_classes])
        y_pred_tree = ops.where(self.valid_mask, y_pred_tree, y_pred.dtype.min)

        y_pred_tree = ops.argmax(y_pred_tree, axis=-1)
        y_match_tree = ops.cast(y_true_tree == y_pred_tree, self.dtype)
        y_valid_tree = ops.cast(y_true_tree != -1, self.dtype)

        y_match_tree = ops.split(y_match_tree, self.num_levels, axis=-1)
        c_match_tree = y_match_tree[:1]
        for i in range(1, self.num_levels):
            c_match_tree.append(
                ops.minimum(c_match_tree[i - 1], y_match_tree[i])
            )
        y_match_tree = ops.concatenate(c_match_tree, axis=-1)

        values = ops.sum(y_match_tree, axis=-1) / ops.sum(y_valid_tree, axis=-1)
        if sample_weight is not None:
            values *= sample_weight
        self.total.assign_add(ops.sum(values))

        if sample_weight is None:
            num_values = ops.cast(ops.size(values), self.dtype)
        else:
            num_values = ops.sum(sample_weight)
        self.count.assign_add(num_values)

    def result(self):
        return ops.divide_no_nan(self.total, self.count)

    def get_config(self):
        config = super().get_config()
        config.update({"tree_paths": self.tree_paths})

        return config

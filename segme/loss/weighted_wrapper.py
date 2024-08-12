from keras.src import ops
from keras.src import tree
from keras.src.losses import LossFunctionWrapper
from keras.src.losses.loss import reduce_weighted_values
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Loss")
class WeightedLossFunctionWrapper(LossFunctionWrapper):
    def __call__(self, y_true, y_pred, sample_weight=None):
        mask = getattr(y_pred, "_keras_mask", None)

        with ops.name_scope(self.name):
            y_pred = tree.map_structure(
                lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_pred
            )
            y_true = tree.map_structure(
                lambda x: ops.convert_to_tensor(x, dtype=self.dtype), y_true
            )
            if sample_weight is not None:
                sample_weight = tree.map_structure(
                    lambda x: ops.convert_to_tensor(x, dtype=self.dtype),
                    sample_weight,
                )

            losses = self.call(y_true, y_pred, sample_weight)

            return reduce_weighted_values(
                losses,
                sample_weight=None,
                mask=mask,
                reduction=self.reduction,
                dtype=self.dtype,
            )

    def call(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)

        if sample_weight is not None:
            y_pred_rank = len(y_pred.shape)
            sample_weight_rank = len(sample_weight.shape)

            if y_pred_rank == sample_weight_rank + 1:
                sample_weight = ops.expand_dims(sample_weight, axis=-1)

        return self.fn(
            y_true, y_pred, sample_weight=sample_weight, **self._fn_kwargs
        )

import tensorflow as tf
from tf_keras import backend
from tf_keras.saving import register_keras_serializable
from tf_keras.src.losses import LossFunctionWrapper
from tf_keras.src.utils import losses_utils, tf_utils


@register_keras_serializable(package='SegMe>Loss')
class WeightedLossFunctionWrapper(LossFunctionWrapper):
    def call(self, y_true, y_pred, sample_weight=None):
        if tf.is_tensor(y_pred) and tf.is_tensor(y_true):
            if sample_weight is None:
                y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
            else:
                y_pred, y_true, sample_weight = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true, sample_weight)

        ag_fn = tf.__internal__.autograph.tf_convert(
            self.fn, tf.__internal__.autograph.control_status_ctx())

        return ag_fn(y_true, y_pred, sample_weight, **self._fn_kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        # If we are wrapping a lambda function strip '<>' from the name as it is not accepted in scope name.
        graph_ctx = tf_utils.graph_context_for_symbolic_tensors(y_true, y_pred, sample_weight)
        with backend.name_scope(self._name_scope), graph_ctx:
            if tf.executing_eagerly():
                call_fn = self.call
            else:
                call_fn = tf.__internal__.autograph.tf_convert(
                    self.call, tf.__internal__.autograph.control_status_ctx())

            # Rescaling disabled, should be done in loss function with weighted_loss(...) call.
            # pred_mask = losses_utils.get_mask(y_pred)
            # sample_weight = losses_utils.apply_valid_mask(y_pred, sample_weight, pred_mask, self._get_reduction())

            losses = call_fn(y_true, y_pred, sample_weight)

            return losses_utils.compute_weighted_loss(losses, None, reduction=self._get_reduction())

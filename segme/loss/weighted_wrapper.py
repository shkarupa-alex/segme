from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_util
from keras import backend, losses
from keras.utils import losses_utils
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import graph_context_for_symbolic_tensors


@register_keras_serializable(package='SegMe')
class WeightedLossFunctionWrapper(losses.LossFunctionWrapper):
    def call(self, y_true, y_pred, sample_weight=None):
        if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
            if sample_weight is None:
                y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
            else:
                y_pred, y_true, sample_weight = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true, sample_weight)
        ag_fn = autograph.tf_convert(self.fn, ag_ctx.control_status_ctx())

        return ag_fn(y_true, y_pred, sample_weight, **self._fn_kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        # If we are wrapping a lambda function strip '<>' from the name as it is not accepted in scope name.
        graph_ctx = graph_context_for_symbolic_tensors(y_true, y_pred, sample_weight)
        with backend.name_scope(self._name_scope), graph_ctx:
            if context.executing_eagerly():
                call_fn = self.call
            else:
                call_fn = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
            losses = call_fn(y_true, y_pred, sample_weight)

            return losses_utils.compute_weighted_loss(losses, None, reduction=self._get_reduction())

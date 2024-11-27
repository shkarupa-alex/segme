from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import compute_gradient
from segme.loss.common_loss import validate_input
from segme.loss.common_loss import weighted_loss
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package="SegMe>Loss")
class HardGradientMeanAbsoluteError(WeightedLossFunctionWrapper):
    """Proposed in: 'Learning-based Sampling for Natural Image Matting'

    Implements Equation [7] in https://openaccess.thecvf.com/
    content_CVPR_2019/papers/
    Tang_Learning-Based_Sampling_for_Natural_Image_Matting_CVPR_2019_paper.pdf
    """

    def __init__(
        self,
        reduction="sum_over_batch_size",
        name="hard_gradient_mean_absolute_error",
    ):
        super().__init__(
            hard_gradient_mean_absolute_error, reduction=reduction, name=name
        )


def hard_gradient_mean_absolute_error(y_true, y_pred, sample_weight):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=4, channel="same"
    )

    g_true_x = compute_gradient(y_true, 1, "sub")
    g_true_x = ops.stop_gradient(g_true_x)
    g_true_y = compute_gradient(y_true, 2, "sub")
    g_true_y = ops.stop_gradient(g_true_y)

    g_pred_x = compute_gradient(y_pred, 1, "sub")
    g_pred_y = compute_gradient(y_pred, 2, "sub")

    g_weight_x, g_weight_y = None, None
    if sample_weight is not None:
        g_weight_x = compute_gradient(sample_weight, 1, "min")
        g_weight_x = ops.stop_gradient(g_weight_x)
        g_weight_y = compute_gradient(sample_weight, 2, "min")
        g_weight_y = ops.stop_gradient(g_weight_y)

    loss = weighted_loss(
        ops.abs(g_true_x - g_pred_x), g_weight_x
    ) + weighted_loss(ops.abs(g_true_y - g_pred_y), g_weight_y)

    return loss

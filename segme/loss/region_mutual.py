from keras.src import backend
from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.loss.common_loss import to_1hot
from segme.loss.common_loss import to_probs
from segme.loss.common_loss import validate_input
from segme.loss.weighted_wrapper import WeightedLossFunctionWrapper
from segme.ops import logdet


@register_keras_serializable(package="SegMe>Loss")
class RegionMutualInformationLoss(WeightedLossFunctionWrapper):
    """Proposed in: 'Region Mutual Information Loss for Semantic Segmentation'

    Implements right sum part in equation [16] in
    https://arxiv.org/pdf/1910.12037
    """

    def __init__(
        self,
        rmi_radius=3,
        pool_way="avgpool",
        pool_stride=4,
        from_logits=False,
        label_smoothing=0.0,
        force_binary=False,
        reduction="sum_over_batch_size",
        name="region_mutual_information_loss",
    ):
        super().__init__(
            region_mutual_information_loss,
            reduction=reduction,
            name=name,
            rmi_radius=rmi_radius,
            pool_way=pool_way,
            pool_stride=pool_stride,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            force_binary=force_binary,
        )


def region_mutual_information_loss(
    y_true,
    y_pred,
    sample_weight,
    rmi_radius,
    pool_stride,
    pool_way,
    from_logits,
    label_smoothing,
    force_binary,
):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype="int64", rank=4, channel="sparse"
    )
    if not 1 <= rmi_radius <= 10:
        raise ValueError("Unsupported RMI radius: {}".format(rmi_radius))
    if pool_stride > 1 and pool_way not in {"maxpool", "avgpool", "resize"}:
        raise ValueError("Unsupported RMI pooling way: {}".format(pool_way))

    y_pred, from_logits = to_probs(
        y_pred, from_logits, force_binary=force_binary
    )

    # Decouple sample_weight to batch items weight and erase invalid pixels
    if sample_weight is not None:
        valid_mask = sample_weight > 0.0
        valid_weight = ops.cast(valid_mask, y_pred.dtype)

        batch_weight = ops.sum(sample_weight, axis=[1, 2, 3]) / (
            ops.sum(valid_weight, axis=[1, 2, 3]) + backend.epsilon()
        )

        y_true *= ops.cast(valid_mask, y_true.dtype)
        y_pred *= valid_weight
    else:
        batch_weight = None
        valid_weight = None

    y_true, y_pred = to_1hot(y_true, y_pred, from_logits, dtype=y_pred.dtype)

    if label_smoothing:
        num_classes = 2 if force_binary else y_true.shape[-1]
        y_true = (
            y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
        )
        if valid_weight:
            y_true *= valid_weight

    # Get region mutual information
    rmi_loss = _rmi_lower_bound(
        y_true,
        y_pred,
        pool_stride=pool_stride,
        pool_way=pool_way,
        rmi_radius=rmi_radius,
    )
    if batch_weight is not None:
        rmi_loss *= batch_weight

    return rmi_loss


def _rmi_lower_bound(y_true, y_pred, pool_stride, pool_way, rmi_radius):
    square_radius = rmi_radius**2
    batch, height, width, channel = ops.shape(y_true)

    if pool_stride > 1:
        if "maxpool" == pool_way:
            y_true = ops.max_pool(
                y_true, pool_stride, strides=pool_stride, padding="same"
            )
            y_pred = ops.max_pool(
                y_pred, pool_stride, strides=pool_stride, padding="same"
            )
        elif "avgpool" == pool_way:
            y_true = ops.average_pool(
                y_true, pool_stride, strides=pool_stride, padding="same"
            )
            y_pred = ops.average_pool(
                y_pred, pool_stride, strides=pool_stride, padding="same"
            )
        elif "resize" == pool_way:  # interpolation
            new_size = height // pool_stride, width // pool_stride
            # NEAREST_NEIGHBOR applied for y_true in reference implementation
            y_true = ops.image.resize(
                y_true, new_size, interpolation="bilinear"
            )
            y_pred = ops.image.resize(
                y_pred, new_size, interpolation="bilinear"
            )
        else:
            raise NotImplementedError(
                "RMI pool way is unknown: {}".format(pool_way)
            )

    # Convert to NCHW for later multiplications
    y_true = ops.transpose(y_true, [0, 3, 1, 2])
    y_pred = ops.transpose(y_pred, [0, 3, 1, 2])

    # Combine the high dimension points from label and probability map.
    # New shape [N, C, radius^2, H, W]
    la_vectors, pr_vectors = _map_get_pairs(y_true, y_pred, radius=rmi_radius)

    la_vectors = ops.reshape(la_vectors, [batch, channel, square_radius, -1])
    la_vectors = ops.cast(la_vectors, "float64")
    la_vectors = ops.stop_gradient(
        la_vectors
    )  # We do not need the gradient of label.
    pr_vectors = ops.reshape(pr_vectors, [batch, channel, square_radius, -1])
    pr_vectors = ops.cast(pr_vectors, "float64")

    # Small diagonal matrix, shape = [1, 1, radius^2, radius^2]
    diag_matrix = ops.eye(square_radius, dtype=pr_vectors.dtype)[
        None, None, ...
    ]
    # Add this factor to ensure the AA^T is positive definite
    diag_matrix *= 5e-4

    # The mean and covariance of these high dimension points
    # Var(X) = E(X^2) - E(X) E(X), N * Var(X) = X^2 - X E(X)
    la_vectors -= ops.mean(la_vectors, axis=-1, keepdims=True)
    la_cov = ops.matmul(la_vectors, ops.moveaxis(la_vectors, -1, -2))
    pr_vectors -= ops.mean(pr_vectors, axis=-1, keepdims=True)
    pr_vectors_t = ops.moveaxis(pr_vectors, -1, -2)
    pr_cov = ops.matmul(pr_vectors, pr_vectors_t)
    pr_cov_inv = ops.inv(pr_cov + diag_matrix)
    la_pr_cov = ops.matmul(la_vectors, pr_vectors_t)

    # The approxiamation of the variance, det(c A) = c^n det(A),
    # A is in n x n shape; then log det(c A) = n log(c) + log det(A).
    # appro_var = appro_var / n_points, we do not divide the appro_var by
    # number of points here, and the purpose is to avoid underflow issue.
    # If A = A^T, A^-1 = (A^-1)^T.
    appro_var = la_cov - ops.matmul(
        ops.matmul(la_pr_cov, pr_cov_inv), ops.moveaxis(la_pr_cov, -1, -2)
    )

    # The lower bound. If A is nonsingular, ln( det(A) ) = Tr( ln(A) ).
    rmi_loss = 0.5 * logdet(appro_var + diag_matrix)
    rmi_loss = ops.cast(rmi_loss, y_pred.dtype)

    # In source: sum over classes, mean over batch samples.
    rmi_loss = ops.mean(rmi_loss, axis=-1) / float(square_radius)

    return rmi_loss


def _map_get_pairs(target, output, radius):
    height, width = ops.shape(target)[2:4]
    new_height, new_width = height - radius + 1, width - radius + 1

    la_ns, pr_ns = [], []
    for y in range(radius):
        for x in range(radius):
            la_now = target[:, :, y : y + new_height, x : x + new_width]
            pr_now = output[:, :, y : y + new_height, x : x + new_width]
            la_ns.append(la_now)
            pr_ns.append(pr_now)

    la_vectors = ops.stack(la_ns, axis=2)
    pr_vectors = ops.stack(pr_ns, axis=2)

    return la_vectors, pr_vectors

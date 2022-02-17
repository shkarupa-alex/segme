import tensorflow as tf
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
from .common_loss import validate_input, to_probs, to_1hot
from .weighted_wrapper import WeightedLossFunctionWrapper


@register_keras_serializable(package='SegMe')
class RegionMutualInformationLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Region Mutual Information Loss for Semantic Segmentation'

    Implements right sum part in equation [16] in https://arxiv.org/pdf/1910.12037.pdf
    """

    def __init__(
            self, from_logits=False, rmi_radius=3, pool_way='avgpool', pool_stride=4,
            reduction=Reduction.AUTO, name='region_mutual_information_loss'):
        super().__init__(
            region_mutual_information_loss, reduction=reduction, name=name, from_logits=from_logits,
            rmi_radius=rmi_radius, pool_way=pool_way, pool_stride=pool_stride)


def region_mutual_information_loss(y_true, y_pred, sample_weight, rmi_radius, pool_stride, pool_way, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype='int32', rank=4, channel='sparse')
    if not 1 <= rmi_radius <= 10:
        raise ValueError('Unsupported RMI radius: {}'.format(rmi_radius))
    if pool_stride > 1 and pool_way not in {'maxpool', 'avgpool', 'resize'}:
        raise ValueError('Unsupported RMI pooling way: {}'.format(pool_way))

    # Decouple sample_weight to batch items weight and erase invalid pixels
    if sample_weight is not None:
        valid_mask = sample_weight > 0.
        valid_weight = tf.cast(valid_mask, y_pred.dtype)

        axis_hw = list(range(1, sample_weight.shape.ndims - 1))
        batch_weight = tf.math.divide_no_nan(
            tf.reduce_sum(sample_weight, axis=axis_hw), tf.reduce_sum(valid_weight, axis=axis_hw))

        y_true = tf.where(valid_mask, y_true, tf.zeros_like(y_true))
        y_pred = tf.where(valid_mask, y_pred, tf.zeros_like(y_pred))
    else:
        batch_weight = None

    y_pred, from_logits = to_probs(y_pred, from_logits, force_sigmoid=True), False
    y_true, y_pred = to_1hot(y_true, y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    # Get region mutual information
    rmi_loss = _rmi_lower_bound(
        y_true, y_pred, batch_weight=batch_weight, pool_stride=pool_stride, pool_way=pool_way, rmi_radius=rmi_radius)

    return rmi_loss


def _rmi_lower_bound(y_true, y_pred, batch_weight, pool_stride, pool_way, rmi_radius):
    square_radius = rmi_radius ** 2
    batch, height, width, channel = tf.unstack(tf.shape(y_true))

    if pool_stride > 1:
        if 'maxpool' == pool_way:
            y_true = tf.nn.max_pool2d(y_true, ksize=pool_stride, strides=pool_stride, padding='SAME')
            y_pred = tf.nn.max_pool2d(y_pred, ksize=pool_stride, strides=pool_stride, padding='SAME')
        elif 'avgpool' == pool_way:
            y_true = tf.nn.avg_pool2d(y_true, ksize=pool_stride, strides=pool_stride, padding='SAME')
            y_pred = tf.nn.avg_pool2d(y_pred, ksize=pool_stride, strides=pool_stride, padding='SAME')
        elif 'resize' == pool_way:  # interpolation
            new_size = height // pool_stride, width // pool_stride
            y_true = tf.image.resize(y_true, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            y_pred = tf.image.resize(y_pred, new_size, method=tf.image.ResizeMethod.BILINEAR)
        else:
            raise NotImplementedError('RMI pool way is unknown: {}'.format(pool_way))

    # Convert to NCHW for later multiplications
    y_true = tf.transpose(y_true, [0, 3, 1, 2])
    y_pred = tf.transpose(y_pred, [0, 3, 1, 2])

    # Combine the high dimension points from label and probability map. New shape [N, C, radius^2, H, W]
    la_vectors, pr_vectors = _map_get_pairs(y_true, y_pred, radius=rmi_radius)

    la_vectors = tf.reshape(la_vectors, [batch, channel, square_radius, -1])
    la_vectors = tf.cast(la_vectors, 'float64')
    la_vectors = tf.stop_gradient(la_vectors)  # We do not need the gradient of label.
    pr_vectors = tf.reshape(pr_vectors, [batch, channel, square_radius, -1])
    pr_vectors = tf.cast(pr_vectors, 'float64')

    # Small diagonal matrix, shape = [1, 1, radius^2, radius^2]
    diag_matrix = tf.eye(square_radius, dtype=pr_vectors.dtype)[None, None, ...]
    # Add this factor to ensure the AA^T is positive definite
    diag_matrix *= 5e-4

    # The mean and covariance of these high dimension points
    # Var(X) = E(X^2) - E(X) E(X), N * Var(X) = X^2 - X E(X)
    la_vectors -= tf.reduce_mean(la_vectors, axis=-1, keepdims=True)
    la_cov = tf.matmul(la_vectors, la_vectors, transpose_b=True)
    pr_vectors -= tf.reduce_mean(pr_vectors, axis=-1, keepdims=True)
    pr_cov = tf.matmul(pr_vectors, pr_vectors, transpose_b=True)
    pr_cov_inv = tf.linalg.inv(pr_cov + diag_matrix)
    la_pr_cov = tf.matmul(la_vectors, pr_vectors, transpose_b=True)

    # The approxiamation of the variance, det(c A) = c^n det(A), A is in n x n shape;
    # then log det(c A) = n log(c) + log det(A).
    # appro_var = appro_var / n_points, we do not divide the appro_var by number of points here,
    # and the purpose is to avoid underflow issue.
    # If A = A^T, A^-1 = (A^-1)^T.
    appro_var = la_cov - tf.matmul(tf.matmul(la_pr_cov, pr_cov_inv), la_pr_cov, transpose_b=True)

    # The lower bound. If A is nonsingular, ln( det(A) ) = Tr( ln(A) ).
    rmi_loss = 0.5 * tf.linalg.logdet(appro_var + diag_matrix)
    rmi_loss = tf.cast(rmi_loss, y_pred.dtype)

    if batch_weight is not None:
        rmi_loss *= batch_weight

    # In source: mean over batch samples, sum over classes.
    # Disable mean to match batch shape
    # rmi_loss = tf.reduce_mean(rmi_loss, axis=0)
    rmi_loss = tf.reduce_sum(rmi_loss / float(square_radius), axis=-1)

    return rmi_loss


def _map_get_pairs(target, output, radius):
    shape = tf.shape(target)
    height, width = shape[2], shape[3]
    new_height, new_width = height - radius + 1, width - radius + 1

    la_ns, pr_ns = [], []
    for y in range(radius):
        for x in range(radius):
            la_now = target[:, :, y:y + new_height, x:x + new_width]
            pr_now = output[:, :, y:y + new_height, x:x + new_width]
            la_ns.append(la_now)
            pr_ns.append(pr_now)

    la_vectors = tf.stack(la_ns, axis=2)
    pr_vectors = tf.stack(pr_ns, axis=2)

    return la_vectors, pr_vectors

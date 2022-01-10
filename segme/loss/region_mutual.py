import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from keras import backend
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import ReductionV2 as Reduction
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
    if not 1 <= rmi_radius <= 10:
        raise ValueError('Unsupported RMI radius: {}'.format(rmi_radius))
    if pool_stride > 1 and pool_way not in {'maxpool', 'avgpool', 'resize'}:
        raise ValueError('Unsupported RMI pooling way: {}'.format(pool_way))

    y_pred = tf.convert_to_tensor(y_pred)

    assert_true_rank = tf.assert_rank(y_true, 4)
    assert_pred_rank = tf.assert_rank(y_pred, 4)

    with tf.control_dependencies([assert_true_rank, assert_pred_rank]):
        epsilon = tf.convert_to_tensor(backend.epsilon(), dtype=y_pred.dtype)

        if y_pred.shape[-1] is None:
            raise ValueError('Number of classes in `y_pred` should be statically known.')

        # In multiclass case replace `softmax` activation with `sigmoid`.
        if not from_logits and y_pred.shape[-1] > 1:
            if hasattr(y_pred, '_keras_logits'):
                y_pred = y_pred._keras_logits
            elif not isinstance(y_pred, (EagerTensor, tf.Variable)) and 'Softmax' != y_pred.op.type:
                # When softmax activation function is used for y_pred operation, we use logits from the softmax
                # function directly to compute loss in order to prevent collapsing zero when training.
                # See b/117284466
                assert len(y_pred.op.inputs) == 1
                y_pred = y_pred.op.inputs[0]
            else:
                raise ValueError('Unable to restore `softmax` logits.')
            from_logits = True

        if from_logits:
            y_pred = tf.nn.sigmoid(y_pred)  # Use sigmoid instead of softmax

        # Label mask -- [N, H, W, 1]
        num_classes = tf.shape(y_pred)[-1]
        y_true_onehot = tf.one_hot(tf.squeeze(y_true, axis=-1), depth=num_classes)
        y_true_onehot = tf.cast(y_true_onehot, dtype=y_pred.dtype)

        # Decouple sample_weight to batch items weight and erase invalid pixels
        if sample_weight is not None:
            axis_hw = list(range(1, sample_weight.shape.ndims - 1))
            batch_weight = tf.reduce_mean(sample_weight, axis=axis_hw)
            valid_pixels = sample_weight > 0.

            y_true_onehot = tf.where(valid_pixels, y_true_onehot, 0)
            y_pred = tf.where(valid_pixels, y_pred, epsilon)
        else:
            batch_weight = None

        # Get region mutual information
        y_true_onehot = tf.stop_gradient(y_true_onehot)

        rmi_loss = _rmi_lower_bound(
            y_true_onehot, y_pred, batch_weight=batch_weight, pool_stride=pool_stride, pool_way=pool_way,
            rmi_radius=rmi_radius)

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

    # Mean over batch samples, sum over classes.
    rmi_loss = tf.reduce_sum(tf.reduce_mean(rmi_loss, axis=0) / float(square_radius))

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

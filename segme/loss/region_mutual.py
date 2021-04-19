import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.keras.utils.control_flow_util import smart_cond
from .weighted_wrapper import WeightedLossFunctionWrapper


@tf.keras.utils.register_keras_serializable(package='SegMe')
class RegionMutualInformationLoss(WeightedLossFunctionWrapper):
    """ Proposed in: 'Region Mutual Information Loss for Semantic Segmentation'

    Implements right sum part in equation [16] in https://arxiv.org/pdf/1910.12037.pdf
    """

    def __init__(
            self, from_logits=False, rmi_radius=3, pool_way='avgpool', pool_stride=4,
            reduction=tf.keras.losses.Reduction.AUTO, name='region_mutual_information_loss'):
        super().__init__(
            region_mutual_information_loss, reduction=reduction, name=name, from_logits=from_logits,
            rmi_radius=rmi_radius, pool_way=pool_way, pool_stride=pool_stride)


def region_mutual_information_loss(y_true, y_pred, sample_weight, rmi_radius, pool_stride, pool_way, from_logits):
    if not 1 <= rmi_radius <= 10:
        raise ValueError('Unsupported RMI radius: {}'.format(rmi_radius))
    if pool_stride > 1 and pool_way not in {'maxpool', 'avgpool', 'resize'}:
        raise ValueError('Unsupported RMI pooling way: {}'.format(pool_way))

    assert_true_rank = tf.assert_rank(y_true, 4)
    assert_pred_rank = tf.assert_rank(y_pred, 4)

    with tf.control_dependencies([assert_true_rank, assert_pred_rank]):
        y_pred = tf.convert_to_tensor(y_pred)
        epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), dtype=y_pred.dtype.base_dtype)

        # Use logits whenever they are available. `softmax` and `sigmoid`
        # activations cache logits on the `output` Tensor.
        if not from_logits:
            if hasattr(y_pred, '_keras_logits'):
                y_pred = y_pred._keras_logits
            elif not isinstance(y_pred, (EagerTensor, tf.Variable)) and y_pred.op.type in {'Sigmoid', 'Softmax'}:
                # When softmax activation function is used for y_pred operation, we
                # use logits from the softmax function directly to compute loss in order
                # to prevent collapsing zero when training.
                # See b/117284466
                assert len(y_pred.op.inputs) == 1
                y_pred = y_pred.op.inputs[0]
            else:
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
                y_pred = smart_cond(
                    tf.shape(y_pred)[-1] == 1,
                    lambda: tf.math.log(y_pred / (1. - y_pred)),  # Restore sigmoid
                    lambda: tf.math.log(y_pred))  # Restore softmax

        # Label mask -- [N, H, W, 1]
        num_classes = tf.shape(y_pred)[-1]
        y_true_onehot = tf.one_hot(tf.squeeze(y_true, axis=-1), depth=num_classes)
        y_true_onehot = tf.cast(y_true_onehot, dtype=y_pred.dtype)
        y_prob = tf.nn.sigmoid(y_pred)

        # Decouple sample_weight to batch items weight and erase invalid pixels
        if sample_weight is not None:
            axis_hw = list(range(1, sample_weight.shape.ndims - 1))
            batch_weight = tf.reduce_mean(sample_weight, axis=axis_hw)
            valid_pixels = sample_weight > 0.

            y_true_onehot = tf.where(valid_pixels, y_true_onehot, 0)
            y_prob = tf.where(valid_pixels, y_prob, epsilon)
        else:
            batch_weight = None

        # Get region mutual information
        y_true_onehot = tf.stop_gradient(y_true_onehot)

        rmi_loss = _rmi_lower_bound(
            y_true_onehot, y_prob, batch_weight=batch_weight, pool_stride=pool_stride, pool_way=pool_way,
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

    # Convert to HCHW for later multiplications
    y_true = tf.transpose(y_true, [0, 3, 1, 2])
    y_pred = tf.transpose(y_pred, [0, 3, 1, 2])

    # Combine the high dimension points from label and probability map. New shape [N, C, radius^2, H, W]
    la_vectors, pr_vectors = _map_get_pairs(y_true, y_pred, radius=rmi_radius)

    la_vectors = tf.reshape(la_vectors, [batch, channel, square_radius, -1])
    la_vectors = tf.stop_gradient(la_vectors)  # We do not need the gradient of label.
    pr_vectors = tf.reshape(pr_vectors, [batch, channel, square_radius, -1])

    # Small diagonal matrix, shape = [1, 1, radius^2, radius^2]
    diag_matrix = tf.eye(square_radius, dtype=y_pred.dtype)[None, None, ...]
    # Add this factor to ensure the AA^T is positive definite
    diag_matrix *= 5e-4  # 1e-8 https://github.com/tensorflow/tensorflow/issues/45235#issuecomment-735917833

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
    try:
        rmi_loss = 0.5 * tf.linalg.logdet(appro_var + diag_matrix)
    except Exception as e:  # TODO
        import numpy as np
        np.save('/tmp/chol_y_true.npy', y_true.numpy())
        np.save('/tmp/chol_y_pred.npy', y_pred.numpy())
        raise e

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

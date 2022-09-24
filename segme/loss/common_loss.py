import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from keras import backend


def validate_input(y_true, y_pred, weight, dtype, rank, channel):
    y_pred = tf.convert_to_tensor(y_pred)

    dtype = y_pred.dtype if dtype is None else dtype
    y_true = tf.cast(y_true, dtype)

    if weight is not None:
        weight = tf.cast(weight, y_pred.dtype)

    if rank is not None and (y_pred.shape.rank != rank or y_true.shape.rank != rank):
        raise ValueError(f'Both labels and predictions must have rank {rank}.')

    if rank is not None and weight is not None and weight.shape.rank != rank:
        raise ValueError(f'Sample weights must have rank {rank}.')

    if y_pred.shape[-1] is None or y_true.shape[-1] is None:
        raise ValueError('Channel dimension of both labels and predictions must be defined.')

    if weight is not None and 1 != weight.shape[-1]:
        raise ValueError('Channel dimension of sample weights muse equals 1.')

    if channel not in {None, 'sparse', 'same'}:
        raise ValueError('Unknown channel size check')

    if 'sparse' == channel and 1 != y_true.shape[-1]:
        raise ValueError('Labels must be sparse-encoded.')

    if 'same' == channel and y_pred.shape[-1] != y_true.shape[-1]:
        raise ValueError('Labels and predictions channel sizes must be equal.')

    return y_true, y_pred, weight


def op_type(y_pred):
    if isinstance(y_pred, (EagerTensor, tf.Variable)):
        return None

    if hasattr(y_pred, '_keras_history'):
        return None

    return y_pred.op.type


def to_logits(y_pred, from_logits):
    logits = None

    # Use logits whenever they are available. `Softmax` and `Sigmoid` activations cache logits on the output tensor
    if hasattr(y_pred, '_keras_logits'):
        logits = y_pred._keras_logits

    # When activation function is used for output operation, use logits from the function directly
    if op_type(y_pred) in {'Sigmoid', 'Softmax'} and 1 == len(y_pred.op.inputs):
        logits = y_pred.op.inputs[0]

    if from_logits and logits is not None:
        raise ValueError('Received `from_logits=True`, but the `y_pred` argument was produced by '
                         'a sigmoid/softmax activation and thus does not represent logits.')

    if from_logits and logits is None:
        return y_pred

    if logits is None:
        raise ValueError('Unable to restore logits.')

    return logits


def to_probs(y_pred, from_logits, force_sigmoid):
    if force_sigmoid and 'Sigmoid' == op_type(y_pred) and 1 == len(y_pred.op.inputs):
        return y_pred

    if force_sigmoid:
        y_pred = to_logits(y_pred, from_logits)
        y_probs = tf.nn.sigmoid(y_pred)
        y_probs._keras_logits = y_pred

        return y_probs

    if not from_logits:
        minmax_range = (tf.reduce_min(y_pred) >= 0.) & (tf.reduce_max(y_pred) <= 1.)
        minmax_assert = tf.assert_equal(
            minmax_range, True, 'Received `from_logits=False`, but the `y_pred` argument has values outside '
                                '[0; 1] range and thus does not represent probabilities.')
        with tf.control_dependencies([minmax_assert]):
            y_probs = tf.identity(y_pred)
            y_probs._keras_logits = y_pred

            return y_probs

    if 1 == y_pred.shape[-1]:
        y_probs = tf.nn.sigmoid(y_pred)
    else:
        y_probs = tf.nn.softmax(y_pred)
    y_probs._keras_logits = y_pred

    return y_probs


def to_1hot(y_true, y_pred):
    if 1 != y_true.shape[-1]:
        raise ValueError('Labels must be sparse-encoded.')

    if 1 == y_pred.shape[-1]:
        assert_min = tf.assert_equal(tf.reduce_min(y_pred) >= 0., True)
        assert_max = tf.assert_equal(tf.reduce_max(y_pred) <= 1., True)
        with tf.control_dependencies([assert_min, assert_max]):
            y_pred = tf.concat([1. - y_pred, y_pred], axis=-1)

    y_true = tf.one_hot(y_true[..., 0], y_pred.shape[-1], dtype=y_true.dtype)

    return y_true, y_pred


def weighted_loss(loss, sample_weight, axis=(1, 2, 3)):
    if sample_weight is not None:
        valid_weight = tf.cast(sample_weight > 0., sample_weight.dtype)
        valid_weight = tf.reduce_mean(valid_weight, axis=[1, 2], keepdims=True)

        sample_weight = tf.math.divide_no_nan(sample_weight, valid_weight)
        sample_weight = tf.stop_gradient(sample_weight)

        loss *= sample_weight

    if axis:
        loss = tf.reduce_mean(loss, axis=axis)

    return loss


def compute_gradient(inputs, axis, reduction):
    if 1 == axis:
        grad = inputs[:, 1:, :, :], inputs[:, :-1, :, :]
    elif 2 == axis:
        grad = inputs[:, :, 1:, :], inputs[:, :, :-1, :]
    else:
        raise ValueError('Unsupported axis: {}'.format(axis))

    if 'sub' == reduction:
        grad = grad[0] - grad[1]
    elif 'min' == reduction:
        grad = tf.minimum(grad[0], grad[1])
    else:
        raise ValueError('Unsupported reduction: {}'.format(reduction))

    return grad


def mae(y_true, y_pred, sample_weight, from_logits, regression=False):
    if regression:
        if from_logits:
            raise ValueError('Regression MAE does not support "from_logits=True"')
        y_true, y_pred, sample_weight = validate_input(
            y_true, y_pred, sample_weight, dtype=None, rank=None, channel='same')
    else:
        y_true, y_pred, sample_weight = validate_input(
            y_true, y_pred, sample_weight, dtype='int32', rank=None, channel='sparse')
        y_pred, from_logits = to_probs(y_pred, from_logits, force_sigmoid=True), False
        y_true, y_pred = to_1hot(y_true, y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

    loss = tf.abs(y_pred - y_true)

    return weighted_loss(loss, sample_weight)


def mse(y_true, y_pred, sample_weight, from_logits, regression=False):
    if regression:
        if from_logits:
            raise ValueError('Regression MAE does not support "from_logits=True"')
        y_true, y_pred, sample_weight = validate_input(
            y_true, y_pred, sample_weight, dtype=None, rank=None, channel='same')
    else:
        y_true, y_pred, sample_weight = validate_input(
            y_true, y_pred, sample_weight, dtype='int32', rank=None, channel='sparse')
        y_pred, from_logits = to_probs(y_pred, from_logits, force_sigmoid=True), False
        y_true, y_pred = to_1hot(y_true, y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

    loss = tf.math.squared_difference(y_pred, y_true)

    return weighted_loss(loss, sample_weight)


def crossentropy(y_true, y_pred, sample_weight, from_logits):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype=None, rank=None, channel='sparse')

    if 1 == y_pred.shape[-1]:
        loss = backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    else:
        loss = backend.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits)[..., None]

    return weighted_loss(loss, sample_weight)


def iou(y_true, y_pred, sample_weight, from_logits, smooth=1., dice=False):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype='int32', rank=4, channel='sparse')
    y_pred, from_logits = to_probs(y_pred, from_logits, force_sigmoid=True), False
    y_true_1h, y_pred_1h = to_1hot(y_true, y_pred)
    y_true_1h = tf.cast(y_true_1h, dtype=y_pred.dtype)

    y_and = y_pred_1h * y_true_1h
    y_or = y_pred_1h + y_true_1h

    # true_pos = y_pred_1h * y_true_1h
    # false_pos = y_pred_1h * (1 - y_true_1h)
    # false_neg = (1 - y_pred_1h) * y_true_1h

    if dice:
        # dice = 2 * true_pos / (2 * true_pos + false_pos + false_neg) = \
        #   2 * y_pred_1h * y_true_1h / (y_pred_1h + y_true_1h)
        numerator = 2. * y_and
        denominator = y_or
    else:
        # iou = true_pos / (true_pos + false_pos + false_neg) = \
        #   y_pred_1h * y_true_1h / (y_pred_1h + y_true_1h - y_pred_1h * y_true_1h)
        numerator = y_and
        denominator = y_or - y_and

    numerator = weighted_loss(numerator, sample_weight, axis=[1, 2])
    denominator = weighted_loss(denominator, sample_weight, axis=[1, 2])

    size = tf.reduce_prod(tf.shape(y_true)[1:3])
    epsilon = smooth / tf.cast(size, y_pred.dtype)

    loss = 1. - (numerator + epsilon) / (denominator + epsilon)

    return tf.reduce_mean(loss, axis=-1)

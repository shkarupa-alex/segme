from keras.src import backend
from keras.src import ops

from segme import backend as back
from segme.ops import squared_difference


def validate_input(y_true, y_pred, weight, dtype, rank, channel):
    y_pred = backend.convert_to_tensor(y_pred)

    dtype = y_pred.dtype if dtype is None else dtype
    y_true = ops.cast(y_true, dtype)

    if weight is not None:
        weight = ops.cast(weight, y_pred.dtype)

    if rank is not None:
        if y_pred.shape.rank != rank:
            raise ValueError(f"Predictions must have rank {rank}.")

        if weight is not None and weight.shape.rank != rank:
            raise ValueError(f"Sample weights must have rank {rank}.")

    if y_pred.shape.rank != y_true.shape.rank:
        raise ValueError("Labels and predictions ranks must be equal.")

    if y_pred.shape[-1] is None or y_true.shape[-1] is None:
        raise ValueError(
            "Channel dimension of both labels and predictions must be defined."
        )

    if weight is not None and 1 != weight.shape[-1]:
        raise ValueError("Channel dimension of sample weights muse equals 1.")

    if channel not in {None, "sparse", "same"}:
        raise ValueError("Unknown channel size check")

    if "sparse" == channel and 1 != y_true.shape[-1]:
        raise ValueError("Labels must be sparse-encoded.")

    if "same" == channel and y_pred.shape[-1] != y_true.shape[-1]:
        raise ValueError("Labels and predictions channel sizes must be equal.")

    return y_true, y_pred, weight


def to_logits(y_pred, from_logits):
    logits = None

    # Use logits whenever they are available. `Softmax` and `Sigmoid`
    # activations cache logits on the output tensor
    if hasattr(y_pred, "_keras_logits"):
        logits = y_pred._keras_logits

    # When activation function is used for output operation, use logits
    # from the function directly
    if back.op_type(y_pred) in {"Sigmoid", "Softmax"} and 1 == len(
        y_pred.op.inputs
    ):
        logits = y_pred.op.inputs[0]

    if from_logits and logits is None:
        return y_pred, True

    if from_logits and logits is not None:
        raise ValueError(
            "Received `from_logits=True`, but the `y_pred` argument was "
            "produced by a sigmoid/softmax activation and thus does not "
            "represent logits."
        )

    if logits is None:
        raise ValueError("Unable to restore logits.")

    return ops.cast(logits, y_pred.dtype), True


def to_probs(y_pred, from_logits, force_binary):
    if force_binary:
        if "Sigmoid" == back.op_type(y_pred) and 1 == len(y_pred.op.inputs):
            if from_logits:
                raise ValueError(
                    "Received `from_logits=True`, but the `y_pred` argument "
                    "was produced by a sigmoid activation and thus does not "
                    "represent logits."
                )

            return y_pred, False

        y_pred, from_logits = to_logits(y_pred, from_logits)
        y_probs = ops.sigmoid(y_pred)
        y_probs._keras_logits = y_pred

        return y_probs, False

    if not from_logits:
        return y_pred, False

    if 1 == y_pred.shape[-1]:
        y_probs = ops.sigmoid(y_pred)
    else:
        y_probs = ops.softmax(y_pred)
    y_probs._keras_logits = y_pred

    return y_probs, False


def to_1hot(y_true, y_pred, from_logits, dtype=None):
    if 1 != y_true.shape[-1]:
        raise ValueError("Labels must be sparse-encoded.")

    if 1 == y_pred.shape[-1]:
        if from_logits:
            y_pred = ops.concatenate([-y_pred, y_pred], axis=-1)
        else:
            y_pred = ops.concatenate([1.0 - y_pred, y_pred], axis=-1)

    y_true = ops.squeeze(y_true, -1)
    y_true = ops.one_hot(y_true, y_pred.shape[-1], dtype=dtype or y_true.dtype)
    y_true = ops.stop_gradient(y_true)

    return y_true, y_pred


def weighted_loss(loss, sample_weight, sample_axes=None, reduce_axes=None):
    if sample_axes is None:
        sample_axes = tuple(range(1, ops.ndim(loss)))
    else:
        bad_axes = set(sample_axes) - set(range(1, ops.ndim(loss)))
        if bad_axes:
            raise ValueError(
                f"Some sample axes can not belong to provided "
                f"inputs: {bad_axes}."
            )

    if reduce_axes is None:
        reduce_axes = sample_axes[:-1]
    else:
        bad_axes = set(reduce_axes) - set(range(1, loss.shape.rank))
        if bad_axes:
            raise ValueError(
                f"Some reduction axes can not belong to provided "
                f"inputs: {bad_axes}."
            )

    if sample_weight is not None:
        if ops.ndim(sample_weight) != ops.ndim(loss):
            raise ValueError("Sample weights and loss ranks must be equal.")

        if 1 != sample_weight.shape[-1]:
            raise ValueError(
                "Channel dimension of sample weights muse equals 1."
            )

        if len(sample_axes) > 1:
            valid_weight = ops.cast(sample_weight > 0.0, sample_weight.dtype)
            valid_weight = ops.mean(
                valid_weight, axis=reduce_axes, keepdims=True
            )

            sample_weight = ops.divide_no_nan(sample_weight, valid_weight)
            sample_weight = ops.stop_gradient(sample_weight)

        loss *= sample_weight

    if sample_axes:
        loss = ops.mean(loss, axis=sample_axes)

    return loss


def compute_gradient(inputs, axis, reduction):
    if 1 == axis:
        grad = inputs[:, 1:, :, :], inputs[:, :-1, :, :]
    elif 2 == axis:
        grad = inputs[:, :, 1:, :], inputs[:, :, :-1, :]
    else:
        raise ValueError("Unsupported axis: {}".format(axis))

    if "sub" == reduction:
        grad = grad[0] - grad[1]
    elif "min" == reduction:
        grad = ops.minimum(grad[0], grad[1])
    else:
        raise ValueError("Unsupported reduction: {}".format(reduction))

    return grad


def smooth_labels(y_true, y_pred, label_smoothing, force_binary):
    y_true = ops.cast(y_true, y_pred.dtype)

    if not label_smoothing:
        return y_true

    if 1 == y_true.shape[-1] and 1 != y_pred.shape[-1]:
        raise ValueError("Labels and predictions channel sizes must be equal.")

    num_classes = 2 if force_binary else max(2, y_pred.shape[-1])
    y_true = y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
    y_true = ops.stop_gradient(y_true)

    return y_true


def mae(
    y_true,
    y_pred,
    sample_weight,
    from_logits,
    regression,
    label_smoothing=0.0,
    force_binary=False,
):
    if regression:
        if from_logits:
            raise ValueError(
                'Regression MAE does not support "from_logits=True"'
            )
        if label_smoothing:
            raise ValueError(
                'Regression MAE does not support "label_smoothing!=0"'
            )
        if force_binary:
            raise ValueError(
                'Regression MAE does not support "label_smoothing!=0"'
            )
        y_true, y_pred, sample_weight = validate_input(
            y_true, y_pred, sample_weight, dtype=None, rank=None, channel="same"
        )
    else:
        y_true, y_pred, sample_weight = validate_input(
            y_true,
            y_pred,
            sample_weight,
            dtype="int64",
            rank=None,
            channel="sparse",
        )
        y_pred, from_logits = to_probs(
            y_pred, from_logits, force_binary=force_binary
        )
        y_true, y_pred = to_1hot(
            y_true, y_pred, from_logits, dtype=y_pred.dtype
        )

        y_true = smooth_labels(y_true, y_pred, label_smoothing, force_binary)

    loss = ops.abs(y_pred - y_true)

    return weighted_loss(loss, sample_weight)


def mse(
    y_true,
    y_pred,
    sample_weight,
    from_logits,
    regression,
    label_smoothing=0.0,
    force_binary=False,
):
    if regression:
        if from_logits:
            raise ValueError(
                'Regression mode does not support "from_logits=True"'
            )
        if label_smoothing:
            raise ValueError(
                'Regression mode does not support "label_smoothing!=0"'
            )
        if force_binary:
            raise ValueError(
                'Regression mode does not support "force_binary=True"'
            )
        y_true, y_pred, sample_weight = validate_input(
            y_true, y_pred, sample_weight, dtype=None, rank=None, channel="same"
        )
    else:
        y_true, y_pred, sample_weight = validate_input(
            y_true,
            y_pred,
            sample_weight,
            dtype="int64",
            rank=None,
            channel="sparse",
        )
        y_pred, from_logits = to_probs(
            y_pred, from_logits, force_binary=force_binary
        )
        y_true, y_pred = to_1hot(
            y_true, y_pred, from_logits, dtype=y_pred.dtype
        )

        y_true = smooth_labels(y_true, y_pred, label_smoothing, force_binary)

    loss = squared_difference(y_pred, y_true)

    return weighted_loss(loss, sample_weight)


def crossentropy(
    y_true,
    y_pred,
    sample_weight,
    from_logits,
    label_smoothing=0.0,
    force_binary=False,
):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype="int64", rank=None, channel=None
    )
    y_pred, from_logits = to_logits(y_pred, from_logits)

    if 1 == y_true.shape[-1] == y_pred.shape[-1]:
        y_true = smooth_labels(y_true, y_pred, label_smoothing, force_binary)
        loss = ops.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    elif 1 == y_true.shape[-1] and 0.0 == label_smoothing and not force_binary:
        loss = ops.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=from_logits
        )[..., None]
    else:
        if 1 == y_true.shape[-1]:
            y_true, y_pred = to_1hot(
                y_true, y_pred, from_logits, dtype=y_pred.dtype
            )

        y_true = smooth_labels(y_true, y_pred, label_smoothing, force_binary)

        if force_binary:
            loss = ops.binary_crossentropy(
                y_true, y_pred, from_logits=from_logits
            )
            loss = ops.sum(loss, axis=-1, keepdims=True)
        else:
            loss = ops.categorical_crossentropy(
                y_true, y_pred, from_logits=from_logits
            )[..., None]

    return weighted_loss(loss, sample_weight)


def iou(
    y_true,
    y_pred,
    sample_weight,
    from_logits,
    smooth=1.0,
    dice=False,
    label_smoothing=0.0,
    force_binary=False,
):
    y_true, y_pred, sample_weight = validate_input(
        y_true, y_pred, sample_weight, dtype="int64", rank=4, channel="sparse"
    )
    y_pred, from_logits = to_probs(
        y_pred, from_logits, force_binary=force_binary
    )
    y_true_1h, y_pred_1h = to_1hot(
        y_true, y_pred, from_logits, dtype=y_pred.dtype
    )

    y_true = smooth_labels(y_true_1h, y_pred_1h, label_smoothing, force_binary)

    y_and = y_pred_1h * y_true_1h
    y_or = y_pred_1h + y_true_1h

    # true_pos = y_pred_1h * y_true_1h
    # false_pos = y_pred_1h * (1 - y_true_1h)
    # false_neg = (1 - y_pred_1h) * y_true_1h

    if dice:
        # dice = 2 * true_pos / (2 * true_pos + false_pos + false_neg) = \
        #   2 * y_pred_1h * y_true_1h / (y_pred_1h + y_true_1h)
        numerator = 2.0 * y_and
        denominator = y_or
    else:
        # iou = true_pos / (true_pos + false_pos + false_neg) = \
        #   y_pred_1h * y_true_1h / (
        #   y_pred_1h + y_true_1h - y_pred_1h * y_true_1h)
        numerator = y_and
        denominator = y_or - y_and

    numerator = weighted_loss(
        numerator, sample_weight, sample_axes=[1, 2], reduce_axes=[1, 2]
    )
    denominator = weighted_loss(
        denominator, sample_weight, sample_axes=[1, 2], reduce_axes=[1, 2]
    )

    size = ops.prod(ops.shape(y_true)[1:-1])
    epsilon = smooth / ops.cast(size, y_pred.dtype)

    loss = 1.0 - (numerator + epsilon) / (denominator + epsilon)

    return ops.mean(loss, axis=-1)

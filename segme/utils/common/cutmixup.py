from keras.src import backend
from keras.src import ops

from segme.ops import convert_image_dtype
from segme.utils.common.augs.common import validate


def cut_mix_up(
    image,
    label,
    weight,
    classes,
    cutmix_alpha=1.0,
    mixup_alpha=0.3,
    cutmix_prob=0.5,
    mixup_prob=0.5,
    name=None,
):
    with backend.name_scope(name or "cut_mix_up"):
        image, _, _ = validate(image, None, None)

        label = backend.convert_to_tensor(label)
        if ops.ndim(label) > 1 and 1 == label.shape[-1]:
            label = ops.squeeze(label, axis=-1)
        if 1 == ops.ndim(label):
            label = ops.one_hot(label, classes)
        label = ops.cast(label, "float32")

        batch = ops.shape(image)[0]
        apply = ops.random.uniform([batch], 0.0, 1.0)
        cutmix_apply = apply < cutmix_prob
        mixup_apply = apply > 1.0 - mixup_prob

        image_, label_, weight_ = cut_mix(
            image, label, weight, cutmix_alpha, classes
        )
        image = ops.where(cutmix_apply[..., None, None, None], image_, image)
        label = ops.where(
            ops.reshape(cutmix_apply, [batch] + [1] * (ops.ndim(label) - 1)),
            label_,
            label,
        )
        if weight is not None:
            weight = ops.where(
                ops.reshape(
                    cutmix_apply, [batch] + [1] * (ops.ndim(label) - 1)
                ),
                weight_,
                weight,
            )

        image_, label_, weight_ = mix_up(
            image, label, weight, mixup_alpha, classes
        )
        image = ops.where(mixup_apply[..., None, None, None], image_, image)
        label = ops.where(
            ops.reshape(mixup_apply, [batch] + [1] * (ops.ndim(label) - 1)),
            label_,
            label,
        )
        if weight is not None:
            weight = ops.where(
                ops.reshape(mixup_apply, [batch] + [1] * (ops.ndim(label) - 1)),
                weight_,
                weight,
            )

        return image, label, weight


def cut_mix(image, label, weight, alpha, classes, name=None):
    with backend.name_scope(name or "cut_mix"):
        image, _, _ = validate(image, None, None)
        label = backend.convert_to_tensor(label)

        if weight is not None:
            weight = backend.convert_to_tensor(weight)
            if ops.ndim(weight) != ops.ndim(label):
                raise ValueError(
                    "Expecting `weight` and `label` ranks to be equal."
                )
            if weight.shape[-1] is None:
                raise ValueError(
                    "Expecting channel dimension of the `weight` to be "
                    "defined. Found `None`."
                )

        batch, height, width, _ = ops.shape(image)

        order = ops.random.shuffle(ops.arange(batch))

        factor = ops.random.gamma((2, batch), alpha=alpha)
        factor = ops.unstack(factor, axis=0)
        factor = factor[0] / sum(factor)

        fullsize = ops.cast(ops.stack([height, width]), "float32")[None]
        cutsize = ops.round(fullsize * ops.sqrt(1.0 - factor[..., None]))
        topleft = ops.round(
            ops.random.uniform(shape=(batch, 2), maxval=fullsize - cutsize)
        )

        return _cut_mix(
            image, label, weight, order, topleft, cutsize, classes, name=None
        )


def _cut_mix(image, label, weight, order, topleft, cutsize, classes, name=None):
    with backend.name_scope(name or "cut_mix_"):
        batch, height, width, _ = ops.shape(image)

        top, left = ops.split(ops.cast(topleft, "int32"), 2, axis=-1)
        bottom, right = ops.split(
            ops.cast(cutsize + topleft, "int32"), 2, axis=-1
        )
        factor = ops.prod(cutsize, axis=-1, keepdims=True) / ops.cast(
            height * width, "float32"
        )

        hrange = ops.repeat(ops.arange(height)[None, :, None], batch, axis=0)
        wrange = ops.repeat(ops.arange(width)[None, None, :], batch, axis=0)
        mask = ((hrange < top[..., None]) | (hrange >= bottom[..., None])) | (
            (wrange < left[..., None]) | (wrange >= right[..., None])
        )

        image_ = ops.take(image, order, axis=0)
        image = ops.where(mask[..., None], image, image_)

        if ops.ndim(label) > 1 and 1 == label.shape[-1]:
            label = ops.squeeze(label, axis=-1)
        if 1 == ops.ndim(label):
            label = ops.one_hot(label, classes)
        label = ops.cast(label, "float32")
        label_ = ops.take(label, order, axis=0)
        label += ops.reshape(factor, [-1] + [1] * (ops.ndim(label) - 1)) * (
            label_ - label
        )

        if weight is not None:
            weight_ = ops.take(weight, order, axis=0)
            weight = weight + ops.reshape(
                factor, [-1] + [1] * (ops.ndim(weight) - 1)
            ) * (weight_ - weight)

        return image, label, weight


def mix_up(image, label, weight, alpha, classes, name=None):
    with backend.name_scope(name or "mix_up"):
        image, _, _ = validate(image, None, None)
        label = backend.convert_to_tensor(label)

        if weight is not None:
            weight = backend.convert_to_tensor(weight)
            if ops.ndim(weight) != ops.ndim(label):
                raise ValueError(
                    "Expecting `weight` and `label` ranks to be equal."
                )
            if weight.shape[-1] is None:
                raise ValueError(
                    "Expecting channel dimension of the `weight` to be "
                    "defined. Found `None`."
                )

        batch = ops.shape(image)[0]

        order = ops.random.shuffle(ops.arange(batch))

        factor = ops.random.gamma((2, batch), alpha=alpha)
        factor = ops.unstack(factor, axis=0)
        factor = factor[0] / sum(factor)
        factor = 0.5 - ops.abs(factor - 0.5)

        return _mix_up(image, label, weight, order, factor, classes, name=None)


def _mix_up(image, label, weight, order, factor, classes, name=None):
    with backend.name_scope(name or "mix_up_"):
        dtype = image.dtype
        image = convert_image_dtype(image, "float32")
        image_ = ops.take(image, order, axis=0)
        image += ops.reshape(factor, [-1] + [1] * (ops.ndim(image) - 1)) * (
            image_ - image
        )
        image = convert_image_dtype(image, dtype, saturate=True)

        if ops.ndim(label) > 1 and 1 == label.shape[-1]:
            label = ops.squeeze(label, axis=-1)
        if 1 == ops.ndim(label):
            label = ops.one_hot(label, classes)
        label = ops.cast(label, "float32")
        label_ = ops.take(label, order, axis=0)
        label += ops.reshape(factor, [-1] + [1] * (ops.ndim(label) - 1)) * (
            label_ - label
        )

        if weight is not None:
            weight = ops.cast(weight, "float32")
            weight_ = ops.take(weight, order, axis=0)
            weight += ops.reshape(
                factor, [-1] + [1] * (ops.ndim(weight) - 1)
            ) * (weight_ - weight)

        return image, label, weight

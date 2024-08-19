import tensorflow as tf

from segme.common.shape import get_shape
from segme.utils.common.augs.common import convert
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
    with tf.name_scope(name or "cut_mix_up"):
        image, _, _ = validate(image, None, None)

        label = tf.convert_to_tensor(label, name="label")
        if label.shape.rank > 1 and 1 == label.shape[-1]:
            label = tf.squeeze(label, axis=-1)
        if 1 == label.shape.rank:
            label = tf.one_hot(label, classes)
        label = tf.cast(label, "float32")

        (batch,), _ = get_shape(image, axis=[0])
        apply = tf.random.uniform([batch], 0.0, 1.0)
        cutmix_apply = apply < cutmix_prob
        mixup_apply = apply > 1.0 - cutmix_prob

        image_, label_, weight_ = cut_mix(
            image, label, weight, cutmix_alpha, classes
        )
        image = tf.where(cutmix_apply[..., None, None, None], image_, image)
        label = tf.where(
            tf.reshape(cutmix_apply, [batch] + [1] * (label.shape.rank - 1)),
            label_,
            label,
        )
        if weight is not None:
            weight = tf.where(
                tf.reshape(
                    cutmix_apply, [batch] + [1] * (label.shape.rank - 1)
                ),
                weight_,
                weight,
            )

        image_, label_, weight_ = mix_up(
            image, label, weight, cutmix_alpha, classes
        )
        image = tf.where(mixup_apply[..., None, None, None], image_, image)
        label = tf.where(
            tf.reshape(mixup_apply, [batch] + [1] * (label.shape.rank - 1)),
            label_,
            label,
        )
        if weight is not None:
            weight = tf.where(
                tf.reshape(mixup_apply, [batch] + [1] * (label.shape.rank - 1)),
                weight_,
                weight,
            )

        return image, label, weight


def cut_mix(image, label, weight, alpha, classes, name=None):
    with tf.name_scope(name or "cut_mix"):
        image, _, _ = validate(image, None, None)
        label = tf.convert_to_tensor(label, name="label")

        if weight is not None:
            weight = tf.convert_to_tensor(weight, name="weight")
            if weight.shape.rank != label.shape.rank:
                raise ValueError(
                    "Expecting `weight` and `label` ranks to be equal."
                )
            if weight.shape[-1] is None:
                raise ValueError(
                    "Expecting channel dimension of the `weight` to be "
                    "defined. Found `None`."
                )

        (batch, height, width), _ = get_shape(image, axis=[0, 1, 2])

        order = tf.random.shuffle(tf.range(0, batch))

        factor = tf.random.gamma((2, batch), alpha=alpha)
        factor = tf.unstack(factor, axis=0)
        factor = factor[0] / sum(factor)

        fullsize = tf.cast(tf.stack([height, width]), "float32")[None]
        cutsize = tf.round(fullsize * tf.math.sqrt(1.0 - factor[..., None]))
        topleft = tf.round(
            tf.random.uniform(shape=(batch, 2), maxval=fullsize - cutsize)
        )

        return _cut_mix(
            image, label, weight, order, topleft, cutsize, classes, name=None
        )


def _cut_mix(image, label, weight, order, topleft, cutsize, classes, name=None):
    with tf.name_scope(name or "cut_mix_"):
        (batch, height, width), _ = get_shape(image, axis=[0, 1, 2])

        top, left = tf.split(tf.cast(topleft, "int32"), 2, axis=-1)
        bottom, right = tf.split(
            tf.cast(cutsize + topleft, "int32"), 2, axis=-1
        )
        factor = tf.reduce_prod(cutsize, axis=-1, keepdims=True) / tf.cast(
            height * width, "float32"
        )

        hrange = tf.repeat(tf.range(height)[None, :, None], batch, axis=0)
        wrange = tf.repeat(tf.range(width)[None, None, :], batch, axis=0)
        mask = ((hrange < top[..., None]) | (hrange >= bottom[..., None])) | (
            (wrange < left[..., None]) | (wrange >= right[..., None])
        )

        image_ = tf.gather(image, order)
        image = tf.where(mask[..., None], image, image_)

        if label.shape.rank > 1 and 1 == label.shape[-1]:
            label = tf.squeeze(label, axis=-1)
        if 1 == label.shape.rank:
            label = tf.one_hot(label, classes)
        label = tf.cast(label, "float32")
        label_ = tf.gather(label, order)
        label += tf.reshape(factor, [-1] + [1] * (label.shape.rank - 1)) * (
            label_ - label
        )

        if weight is not None:
            weight_ = tf.gather(weight, order)
            weight = weight + tf.reshape(
                factor, [-1] + [1] * (weight.shape.rank - 1)
            ) * (weight_ - weight)

        return image, label, weight


def mix_up(image, label, weight, alpha, classes, name=None):
    with tf.name_scope(name or "mix_up"):
        image, _, _ = validate(image, None, None)
        label = tf.convert_to_tensor(label, name="label")

        if weight is not None:
            weight = tf.convert_to_tensor(weight, name="weight")
            if weight.shape.rank != label.shape.rank:
                raise ValueError(
                    "Expecting `weight` and `label` ranks to be equal."
                )
            if weight.shape[-1] is None:
                raise ValueError(
                    "Expecting channel dimension of the `weight` to be "
                    "defined. Found `None`."
                )

        (batch,), _ = get_shape(image, axis=[0])

        order = tf.random.shuffle(tf.range(0, batch))

        factor = tf.random.gamma((2, batch), alpha=alpha)
        factor = tf.unstack(factor, axis=0)
        factor = factor[0] / sum(factor)
        factor = 0.5 - tf.abs(factor - 0.5)

        return _mix_up(image, label, weight, order, factor, classes, name=None)


def _mix_up(image, label, weight, order, factor, classes, name=None):
    with tf.name_scope(name or "mix_up_"):
        dtype = image.dtype
        image = convert(image, "float32")
        image_ = tf.gather(image, order)
        image += tf.reshape(factor, [-1] + [1] * (image.shape.rank - 1)) * (
            image_ - image
        )
        image = convert(image, dtype, saturate=True)

        if label.shape.rank > 1 and 1 == label.shape[-1]:
            label = tf.squeeze(label, axis=-1)
        if 1 == label.shape.rank:
            label = tf.one_hot(label, classes)
        label = tf.cast(label, "float32")
        label_ = tf.gather(label, order)
        label += tf.reshape(factor, [-1] + [1] * (label.shape.rank - 1)) * (
            label_ - label
        )

        if weight is not None:
            weight = tf.cast(weight, "float32")
            weight_ = tf.gather(weight, order)
            weight += tf.reshape(
                factor, [-1] + [1] * (weight.shape.rank - 1)
            ) * (weight_ - weight)

        return image, label, weight

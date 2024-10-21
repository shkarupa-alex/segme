from keras.src import backend
from keras.src import ops
from tensorflow.python.ops import data_flow_ops

from segme.ops import convert_image_dtype


def apply(image, masks, weight, prob, image_fn, mask_fn, weight_fn, name=None):
    def _some_safe(original, aug_fn, condition):
        with backend.name_scope(name or "some_safe"):
            shape = original.shape
            batch = ops.size(condition)
            indices = ops.arange(batch, dtype="int32")
            condition_ = ~condition
            aug_idx, orig_idx = indices[condition], indices[condition_]
            augmented, original = original[condition], original[condition_]
            augmented = aug_fn(augmented)

            result = data_flow_ops.parallel_dynamic_stitch(
                [aug_idx, orig_idx], [augmented, original]
            )
            result.set_shape(shape)

            return result

    def _some(original, aug_fn, condition):
        with backend.name_scope(name or "some"):
            return ops.cond(
                ops.any(condition),
                lambda: _some_safe(original, aug_fn, condition),
                lambda: original,
            )

    def _all(original, aug_fn, condition):
        with backend.name_scope(name or "all"):
            return ops.cond(
                condition,
                lambda: aug_fn(original),
                lambda: original,
            )

    with backend.name_scope(name or "apply"):
        image, masks, weight = validate(image, masks, weight)

        prob = backend.convert_to_tensor(prob)
        if ops.ndim(prob):
            raise ValueError("Expecting `prob` to be a scalar.")

        batch, height, width = ops.shape(image)[:3]
        apply_some = ops.random.uniform([batch]) < prob
        apply_all = ops.random.uniform([]) < prob
        static_size = isinstance(height, int) and isinstance(width, int)
        square_size = (
            height == width if static_size else ops.equal(height, width)
        )

        if static_size and square_size:
            image = _some(image, image_fn, apply_some)
            if masks is not None and mask_fn is not None:
                masks = [_some(m, mask_fn, apply_some) for m in masks]
            if weight is not None and weight_fn is not None:
                weight = _some(weight, weight_fn, apply_some)
        elif static_size and not square_size:
            image = _all(image, image_fn, apply_all)
            if masks is not None and mask_fn is not None:
                masks = [_all(m, mask_fn, apply_all) for m in masks]
            if weight is not None and weight_fn is not None:
                weight = _all(weight, weight_fn, apply_all)
        else:
            image = ops.cond(
                square_size,
                lambda: _some(image, image_fn, apply_some),
                lambda: _all(image, image_fn, apply_all),
            )
            if masks is not None:
                masks = [
                    ops.cond(
                        square_size,
                        lambda: _some(m, mask_fn, apply_some),
                        lambda: _all(m, mask_fn, apply_all),
                    )
                    for m in masks
                ]
            if weight is not None:
                weight = ops.cond(
                    square_size,
                    lambda: _some(weight, weight_fn, apply_some),
                    lambda: _all(weight, weight_fn, apply_all),
                )

        return image, masks, weight


def blend(original, augmented, factor, name=None):
    with backend.name_scope(name or "blend"):
        original, _, _ = validate(original, None, None)
        augmented, _, _ = validate(augmented, None, None)

        dtype = original.dtype
        original = convert_image_dtype(original, "float32")
        augmented = convert_image_dtype(augmented, "float32")

        blended = original + factor * (augmented - original)

        return convert_image_dtype(blended, dtype, saturate=True)


def wrap(image, name=None):
    with backend.name_scope(name or "wrap"):
        image, _, _ = validate(image, None, None)

        image = ops.pad(image, [(0, 0)] * 3 + [(0, 1)], constant_values=1)

        return image


def unwrap(image, replace=None, name=None):
    with backend.name_scope(name or "wrap"):
        image, _, _ = validate(image, None, None)

        image, mask = ops.split(image, [image.shape[-1] - 1], axis=-1)

        if replace is not None:
            replace = backend.convert_to_tensor(replace, image.dtype)
            replace, _, _ = validate(replace, None, None)
        else:
            replace = ops.mean(image, axis=[1, 2], keepdims=True)

        image = ops.where(ops.equal(mask, 1), image, replace)

        return image


def validate(image, masks, weight, name=None):
    with backend.name_scope(name or "validate"):
        image_ = backend.convert_to_tensor(image)
        if 4 != ops.ndim(image_):
            raise ValueError("Expecting `image` rank to be 4.")
        if image_.shape[-1] is None:
            raise ValueError(
                "Expecting channel dimension of the `image` to be defined. "
                "Found `None`."
            )

        if masks is not None:
            masks_ = []
            for i, mask in enumerate(masks):
                mask_ = backend.convert_to_tensor(mask, dtype=mask.dtype)
                if 4 != ops.ndim(mask_):
                    raise ValueError("Expecting `masks` items rank to be 4.")
                if mask_.shape[-1] is None:
                    raise ValueError(
                        "Expecting channel dimension of the `mask` to be "
                        "defined. Found `None`."
                    )
                masks_.append(mask_)
        else:
            masks_ = None

        if weight is not None:
            weight_ = backend.convert_to_tensor(weight)
            if 4 != ops.ndim(weight_):
                raise ValueError("Expecting `weight` rank to be 4.")
            if weight_.shape[-1] is None:
                raise ValueError(
                    "Expecting channel dimension of the `weight` to be "
                    "defined. Found `None`."
                )
        else:
            weight_ = None

        return image_, masks_, weight_

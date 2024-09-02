from keras.src import backend
from keras.src import ops
from tensorflow.python.ops import data_flow_ops

from segme.ops import convert_image_dtype


def apply(image, masks, weight, prob, image_fn, mask_fn, weight_fn, name=None):
    def _some(original, condition, aug_fn, orig_idx, aug_idx):
        with backend.name_scope(name or "some"):
            shape = original.shape

            augmented, original = original[condition], original[~condition]
            augmented = aug_fn(augmented)

            result = data_flow_ops.parallel_dynamic_stitch(
                [aug_idx, orig_idx], [augmented, original]
            )
            result.set_shape(shape)

            return result

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
        if prob.shape.rank:
            raise ValueError("Expecting `prob` to be a scalar.")

        batch, height, width, _ = ops.shape(image)
        static_size = isinstance(height, int) and isinstance(width, int)
        square_size = (
            height == width if static_size else ops.equal(height, width)
        )

        switch_some = ops.random.uniform([batch]) < prob
        switch_all = switch_some[0]

        if static_size and square_size:
            indices = ops.arange(batch, dtype="int32")
            indices_, indices = indices[switch_some], indices[~switch_some]

            image = _some(image, switch_some, image_fn, indices, indices_)

            if masks is not None and mask_fn is not None:
                masks = [
                    _some(m, switch_some, mask_fn, indices, indices_)
                    for m in masks
                ]

            if weight is not None and weight_fn is not None:
                weight = _some(
                    weight, switch_some, weight_fn, indices, indices_
                )

        elif static_size and not square_size:
            image = _all(image, image_fn, switch_all)

            if masks is not None and mask_fn is not None:
                masks = [_all(m, mask_fn, switch_all) for m in masks]

            if weight is not None and weight_fn is not None:
                weight = _all(weight, weight_fn, switch_all)

        else:
            indices = ops.arange(batch, dtype="int32")
            indices_, indices = indices[switch_some], indices[~switch_some]

            image = ops.cond(
                square_size,
                lambda: _some(image, switch_some, image_fn, indices, indices_),
                lambda: _all(image, image_fn, switch_all),
            )

            if masks is not None:
                temp = []
                for mask in masks:
                    mask = ops.cond(
                        square_size,
                        lambda: _some(
                            mask, switch_some, mask_fn, indices, indices_
                        ),
                        lambda: _all(mask, mask_fn, switch_all),
                    )
                    temp.append(mask)
                masks = temp

            if weight is not None:
                weight = ops.cond(
                    square_size,
                    lambda: _some(
                        weight, switch_some, weight_fn, indices, indices_
                    ),
                    lambda: _all(weight, weight_fn, switch_all),
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
        if 4 != image_.shape.rank:
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
                if 4 != mask_.shape.rank:
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
            if 4 != weight_.shape.rank:
                raise ValueError("Expecting `weight` rank to be 4.")
            if weight_.shape[-1] is None:
                raise ValueError(
                    "Expecting channel dimension of the `weight` to be "
                    "defined. Found `None`."
                )
        else:
            weight_ = None

        return image_, masks_, weight_

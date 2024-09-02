import numpy as np
from keras.src import backend
from keras.src import ops

from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def erase(image, masks, weight, prob, area, replace=None, name=None):
    with backend.name_scope(name or "erase"):
        image, masks, weight = validate(image, masks, weight)

        if isinstance(area, float):
            area = [0.0, area]
        area = backend.convert_to_tensor(area, "float32")
        area = ops.unstack(ops.sqrt(area) / 2.0)

        batch, height, width, _ = ops.shape(image)
        height = ops.cast(height, "float32")
        width = ops.cast(width, "float32")

        hcenter = ops.random.uniform([batch, 1, 1, 1], maxval=height)
        wcenter = ops.random.uniform([batch, 1, 1, 1], maxval=width)

        hradius = ops.random.uniform(
            [batch, 1, 1, 1], minval=area[0] * height, maxval=area[1] * height
        )
        wradius = ops.random.uniform(
            [batch, 1, 1, 1], minval=area[0] * width, maxval=area[1] * width
        )

        hrange = ops.repeat(
            ops.arange(height)[None, :, None, None], batch, axis=0
        )
        wrange = ops.repeat(
            ops.arange(width)[None, None, :, None], batch, axis=0
        )

        mask = ((hrange < hcenter - hradius) | (hrange > hcenter + hradius)) | (
            (wrange < wcenter - wradius) | (wrange > wcenter + wradius)
        )

        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _erase(x, mask, replace=replace),
            None,
            lambda x: _erase(x, mask, replace=np.zeros([1, 1, 1, x.shape[-1]])),
        )


def _erase(image, mask, replace=None, name=None):
    with backend.name_scope(name or "erase_"):
        image, _, mask = validate(image, None, mask)

        batch = ops.shape(image)[0]
        mask = mask[:batch]

        if replace is not None:
            replace = backend.convert_to_tensor(replace, image.dtype)
            replace, _, _ = validate(replace, None, None)
        else:
            replace = ops.mean(image, axis=[1, 2], keepdims=True)

        image = ops.where(mask, image, replace)

        return image

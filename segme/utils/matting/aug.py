from keras.src import backend
from keras.src import ops

from segme.ops import convert_image_dtype


def augment_alpha(alpha, prob=0.3, max_pow=2.0, seed=None):
    with backend.name_scope("augment_alpha"):
        alpha = backend.convert_to_tensor(alpha, "uint8")

        if 4 != ops.ndim(alpha):
            raise ValueError("Expecting `alpha` rank to be 4.")

        if 1 != alpha.shape[-1]:
            raise ValueError("Expecting `alpha` channels size to be 1.")

        if "uint8" != alpha.dtype:
            raise ValueError("Expecting `alpha` dtype to be `uint8`.")

        alpha = convert_image_dtype(alpha, "float32")

        batch = ops.shape(alpha)[0]
        gamma, direction, apply, invert = ops.split(
            ops.random.uniform([batch, 1, 1, 4], 0.0, 1.0, seed=seed),
            4,
            axis=-1,
        )

        direction = ops.cast(direction > 0.5, direction.dtype)
        gamma = gamma * (max_pow - 1.0) + 1.0
        gamma = direction * gamma + (1.0 - direction) / gamma

        invert = ops.cast(invert > 0.5, invert.dtype)
        alpha_ = (1.0 - alpha) * invert + alpha * (1.0 - invert)
        alpha_ = ops.power(alpha_, gamma)
        alpha_ = (1.0 - alpha_) * invert + alpha_ * (1.0 - invert)

        apply = ops.cast(apply < prob, alpha.dtype)
        alpha = alpha_ * apply + alpha * (1 - apply)

        return convert_image_dtype(alpha, "uint8", saturate=True)


def augment_trimap(trimap, prob=0.1, seed=None):
    with backend.name_scope("augment_trimap"):
        trimap = backend.convert_to_tensor(trimap, "uint8")

        if 4 != ops.ndim(trimap):
            raise ValueError("Expecting `trimap` rank to be 4.")

        if 1 != trimap.shape[-1]:
            raise ValueError("Expecting `trimap` channels size to be 1.")

        if "uint8" != trimap.dtype:
            raise ValueError("Expecting `trimap` dtype to be `uint8`.")

        batch = ops.shape(trimap)[0]
        apply = ops.random.uniform([batch, 1, 1, 1], 0.0, 1.0, seed=seed)
        apply = ops.cast(apply < prob, trimap.dtype)

        trimap_ = ops.minimum(trimap, 128)

        return trimap_ * apply + trimap * (1 - apply)

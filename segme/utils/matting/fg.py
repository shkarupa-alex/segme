import numpy as np
from keras.src import backend
from keras.src import ops


def _solve_fg_step(fg, da0, da1, da2, da3, denom, term0):
    fg_pad = ops.pad(fg, ((0, 0), (1, 1), (1, 1), (0, 0)), "REFLECT")
    term1 = (
        da0 * fg_pad[:, 1:-1, :-2]
        + da1 * fg_pad[:, 1:-1, 2:]
        + da2 * fg_pad[:, :-2, 1:-1]
        + da3 * fg_pad[:, 2:, 1:-1]
    ) / denom
    fg = term0 + term1

    return fg, da0, da1, da2, da3, denom, term0


def _solve_fg_level(level, alpha, afg, fg, height, width, levels, kappa, steps):
    height_ = ops.cast(ops.round(height ** (level / levels)), "int32")
    width_ = ops.cast(ops.round(width ** (level / levels)), "int32")

    afg_ = ops.image.resize(afg, [height_, width_], interpolation="area")
    afg_ = ops.clip(afg_, 0.0, 1.0)

    alpha_ = ops.image.resize(alpha, [height_, width_], interpolation="area")
    alpha_ = ops.clip(alpha_, 0.0, 1.0)

    fg = ops.image.resize(fg, [height_, width_], interpolation="lanczos5")
    fg = ops.clip(fg, 0.0, 1.0)

    a00 = ops.square(1 - alpha_)

    alpha_pad = ops.pad(alpha_, ((0, 0), (1, 1), (1, 1), (0, 0)), "REFLECT")
    da0 = kappa * (a00 + ops.square(1 - alpha_pad[:, 1:-1, :-2]))
    da1 = kappa * (a00 + ops.square(1 - alpha_pad[:, 1:-1, 2:]))
    da2 = kappa * (a00 + ops.square(1 - alpha_pad[:, :-2, 1:-1]))
    da3 = kappa * (a00 + ops.square(1 - alpha_pad[:, 2:, 1:-1]))

    denom = alpha_**2 + da0 + da1 + da2 + da3
    term0 = alpha_ * afg_ / denom

    steps_ = ops.cast(2 ** ops.minimum(levels - level, 5) * steps, "int32")

    fg, _, _, _, _, _, _ = ops.while_loop(
        lambda *_: True,
        _solve_fg_step,
        (fg, da0, da1, da2, da3, denom, term0),
        maximum_iterations=steps_,
    )

    return level + 1.0, alpha, afg, fg, height, width, levels, kappa, steps


def solve_fg(image, alpha, kappa=1.0, steps=16, name=None):
    with backend.name_scope(name or "solve_fg"):
        image = backend.convert_to_tensor(image, "uint8")
        alpha = backend.convert_to_tensor(alpha, "uint8")
        steps = backend.convert_to_tensor(steps, "float32")

        if 4 != ops.ndim(image):
            raise ValueError("Expecting `image` rank to be 4.")

        if 3 != image.shape[-1]:
            raise ValueError("Expecting `image` channels size to be 3.")

        if 4 != ops.ndim(alpha):
            raise ValueError("Expecting `alpha` rank to be 4.")

        if 1 != alpha.shape[-1]:
            raise ValueError("Expecting `alpha` channels size to be 1.")

        if "uint8" != image.dtype or "uint8" != alpha.dtype:
            raise ValueError(
                "Expecting `image` and `alpha` dtype to be `uint8`."
            )

        image = ops.cast(image, "float32") / 255.0
        alpha = ops.cast(alpha, "float32") / 255.0

        afg = alpha * image

        fg = ops.image.resize(image, (2, 2), interpolation="area")
        fg = ops.clip(fg, 0.0, 1.0)

        height, width = ops.shape(image)[1:3]
        levels = ops.cast(ops.maximum(height, width), "float32")
        levels = ops.ceil(ops.log(levels) / np.log(2))

        _, _, _, fg, _, _, _, _, _ = ops.while_loop(
            lambda level, *_: level <= levels,
            _solve_fg_level,
            [1.0, alpha, afg, fg, height, width, levels, kappa, steps],
        )

        fg = ops.clip(fg, 0.0, 1.0) * 255.0
        fg = ops.cast(ops.round(fg), "uint8")
        fg.set_shape(image.shape)

        return fg

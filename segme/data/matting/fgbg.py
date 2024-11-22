import numpy as np
from keras.src import backend
from keras.src import ops


def _solve_fgbg_step(step, image, fg, bg, alpha, grad_weight, regularization):
    a0 = alpha
    a1 = 1.0 - a0

    a00 = a0**2
    a01 = a0 * a1
    a11 = a1**2

    b0 = a0 * image
    b1 = a1 * image

    pad_a0 = ops.pad(a0, ((0, 0), (1, 1), (1, 1), (0, 0)), "SYMMETRIC")
    da0 = ops.abs(a0 - pad_a0[:, 1:-1, :-2]) * grad_weight + regularization
    da1 = ops.abs(a0 - pad_a0[:, 1:-1, 2:]) * grad_weight + regularization
    da2 = ops.abs(a0 - pad_a0[:, :-2, 1:-1]) * grad_weight + regularization
    da3 = ops.abs(a0 - pad_a0[:, 2:, 1:-1]) * grad_weight + regularization

    da = da0 + da1 + da2 + da3
    a00 += da
    a11 += da

    pad_fg = ops.pad(fg, ((0, 0), (1, 1), (1, 1), (0, 0)), "SYMMETRIC")
    pad_bg = ops.pad(bg, ((0, 0), (1, 1), (1, 1), (0, 0)), "SYMMETRIC")

    b0 += (
        da0 * pad_fg[:, 1:-1, :-2]
        + da1 * pad_fg[:, 1:-1, 2:]
        + da2 * pad_fg[:, :-2, 1:-1]
        + da3 * pad_fg[:, 2:, 1:-1]
    )
    b1 += (
        da0 * pad_bg[:, 1:-1, :-2]
        + da1 * pad_bg[:, 1:-1, 2:]
        + da2 * pad_bg[:, :-2, 1:-1]
        + da3 * pad_bg[:, 2:, 1:-1]
    )

    inv_det = 1.0 / (a00 * a11 - a01**2)
    b00 = a11 * inv_det
    b01 = -a01 * inv_det
    b11 = a00 * inv_det

    fg = ops.clip(b00 * b0 + b01 * b1, 0.0, 1.0)
    bg = ops.clip(b01 * b0 + b11 * b1, 0.0, 1.0)

    return step + 1, image, fg, bg, alpha, grad_weight, regularization


def _solve_fgbg_level(
    level,
    image,
    fg,
    bg,
    alpha,
    height,
    width,
    levels,
    small_size,
    small_steps,
    big_steps,
    grad_weight,
    regularization,
):
    height_ = ops.cast(
        ops.round(ops.cast(height, "float32") ** (level / levels)), "int32"
    )
    width_ = ops.cast(
        ops.round(ops.cast(width, "float32") ** (level / levels)), "int32"
    )

    image_ = ops.image.resize(image, [height_, width_])
    alpha_ = ops.image.resize(alpha, [height_, width_])
    fg = ops.image.resize(fg, [height_, width_])
    bg = ops.image.resize(bg, [height_, width_])

    steps_cond = ops.cast(
        (height_ <= small_size) & (width_ <= small_size), "int32"
    )
    steps = steps_cond * small_steps + (1 - steps_cond) * big_steps

    _, _, fg, bg, _, _, _ = ops.while_loop(
        lambda step, *_: step < steps,
        _solve_fgbg_step,
        (0, image_, fg, bg, alpha_, grad_weight, regularization),
    )

    return (
        level + 1.0,
        image,
        fg,
        bg,
        alpha,
        height,
        width,
        levels,
        small_size,
        small_steps,
        big_steps,
        grad_weight,
        regularization,
    )


def solve_fgbg(
    image,
    alpha,
    regularization=0.005,
    small_size=32,
    small_steps=10,
    big_steps=2,
    grad_weight=0.1,
    name=None,
):
    with backend.name_scope(name or "solve_fgbg"):
        image = backend.convert_to_tensor(image, "uint8")
        alpha = backend.convert_to_tensor(alpha, "uint8")

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

        fg = ops.divide_no_nan(
            ops.sum(image * alpha, axis=[1, 2], keepdims=True),
            ops.sum(alpha, axis=[1, 2], keepdims=True),
        )
        bg = ops.divide_no_nan(
            ops.sum(image * (1.0 - alpha), axis=[1, 2], keepdims=True),
            ops.sum(1.0 - alpha, axis=[1, 2], keepdims=True),
        )

        height, width = ops.shape(image)[1:3]
        levels = ops.cast(ops.maximum(height, width), "float32")
        levels = ops.ceil(ops.log(levels) / np.log(2))

        _, _, fg, bg, _, _, _, _, _, _, _, _, _ = ops.while_loop(
            lambda level, *_: level <= levels,
            _solve_fgbg_level,
            (
                0.0,
                image,
                fg,
                bg,
                alpha,
                height,
                width,
                levels,
                small_size,
                small_steps,
                big_steps,
                grad_weight,
                regularization,
            ),
        )

        fg = ops.cast(ops.round(fg * 255.0), "uint8")
        bg = ops.cast(ops.round(bg * 255.0), "uint8")

        return fg, bg

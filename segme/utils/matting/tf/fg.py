import numpy as np
import tensorflow as tf

from segme.common.shape import get_shape


def _solve_fg_step(fg, da0, da1, da2, da3, denom, term0):
    fg_pad = tf.pad(fg, ((0, 0), (1, 1), (1, 1), (0, 0)), "REFLECT")
    term1 = (
        da0 * fg_pad[:, 1:-1, :-2]
        + da1 * fg_pad[:, 1:-1, 2:]
        + da2 * fg_pad[:, :-2, 1:-1]
        + da3 * fg_pad[:, 2:, 1:-1]
    ) / denom
    fg = term0 + term1

    return fg, da0, da1, da2, da3, denom, term0


def _solve_fg_level(level, alpha, afg, fg, height, width, levels, kappa, steps):
    height_ = tf.cast(tf.math.round(height ** (level / levels)), "int32")
    width_ = tf.cast(tf.math.round(width ** (level / levels)), "int32")

    afg_ = tf.image.resize(
        afg, [height_, width_], method=tf.image.ResizeMethod.AREA
    )
    afg_ = tf.clip_by_value(afg_, 0.0, 1.0)

    alpha_ = tf.image.resize(
        alpha, [height_, width_], method=tf.image.ResizeMethod.AREA
    )
    alpha_ = tf.clip_by_value(alpha_, 0.0, 1.0)

    fg = tf.image.resize(
        fg, [height_, width_], method=tf.image.ResizeMethod.LANCZOS5
    )
    fg = tf.clip_by_value(fg, 0.0, 1.0)

    a00 = (1 - alpha_) ** 2

    alpha_pad = tf.pad(alpha_, ((0, 0), (1, 1), (1, 1), (0, 0)), "REFLECT")
    da0 = kappa * (a00 + (1 - alpha_pad[:, 1:-1, :-2]) ** 2)
    da1 = kappa * (a00 + (1 - alpha_pad[:, 1:-1, 2:]) ** 2)
    da2 = kappa * (a00 + (1 - alpha_pad[:, :-2, 1:-1]) ** 2)
    da3 = kappa * (a00 + (1 - alpha_pad[:, 2:, 1:-1]) ** 2)

    denom = alpha_**2 + da0 + da1 + da2 + da3
    term0 = alpha_ * afg_ / denom

    steps_ = tf.cast(2 ** tf.math.minimum(levels - level, 5) * steps, "int32")

    shapes = (
        [[None, None, None, 3]]
        + [[None, None, None, 1]] * 5
        + [[None, None, None, 3]]
    )
    shapes = tuple(map(tf.TensorShape, shapes))
    fg, _, _, _, _, _, _ = tf.while_loop(
        lambda *_: True,
        _solve_fg_step,
        (fg, da0, da1, da2, da3, denom, term0),
        shape_invariants=shapes,
        maximum_iterations=steps_,
    )

    return level + 1.0, alpha, afg, fg, height, width, levels, kappa, steps


def solve_fg(image, alpha, kappa=1.0, steps=16, name=None):
    with tf.name_scope(name or "solve_fg"):
        image = tf.convert_to_tensor(image, "uint8")
        alpha = tf.convert_to_tensor(alpha, "uint8")
        steps = tf.convert_to_tensor(steps, "float32")

        if 4 != image.shape.rank:
            raise ValueError("Expecting `image` rank to be 4.")

        if 3 != image.shape[-1]:
            raise ValueError("Expecting `image` channels size to be 3.")

        if 4 != image.shape.rank:
            raise ValueError("Expecting `alpha` rank to be 4.")

        if 1 != alpha.shape[-1]:
            raise ValueError("Expecting `alpha` channels size to be 1.")

        if "uint8" != image.dtype or "uint8" != alpha.dtype:
            raise ValueError(
                "Expecting `image` and `alpha` dtype to be `uint8`."
            )

        image = tf.cast(image, "float32") / 255.0
        alpha = tf.cast(alpha, "float32") / 255.0

        afg = alpha * image

        fg = tf.image.resize(image, (2, 2), method=tf.image.ResizeMethod.AREA)
        fg = tf.clip_by_value(fg, 0.0, 1.0)

        (height, width), _ = get_shape(image, axis=[1, 2], dtype="float32")
        levels = tf.cast(tf.math.maximum(height, width), "float32")
        levels = tf.math.ceil(tf.math.log(levels) / np.log(2))

        shapes = (
            [[], [None, None, None, 1]] + [[None, None, None, 3]] * 2 + [[]] * 5
        )
        shapes = list(map(tf.TensorShape, shapes))
        _, _, _, fg, _, _, _, _, _ = tf.while_loop(
            lambda level, *_: level <= levels,
            _solve_fg_level,
            [1.0, alpha, afg, fg, height, width, levels, kappa, steps],
            shape_invariants=shapes,
        )

        fg = tf.clip_by_value(fg, 0.0, 1.0) * 255.0
        fg = tf.cast(tf.round(fg), "uint8")
        fg.set_shape(image.shape)

        return fg

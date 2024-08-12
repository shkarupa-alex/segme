import cv2
import numpy as np


def solve_fgbg(
    image,
    alpha,
    regularization=0.005,
    small_size=32,
    small_steps=10,
    big_steps=2,
    grad_weight=0.1,
):
    # Based on https://github.com/pymatting/pymatting/blob/master/pymatting/foreground/estimate_foreground_ml.py

    if 3 != len(image.shape):
        raise ValueError("Expecting `image` rank to be 3.")

    if 3 != image.shape[-1]:
        raise ValueError("Expecting `image` channels size to be 3.")

    if len(alpha.shape) not in {2, 3}:
        raise ValueError("Expecting `alpha` rank to be 2 or 3.")

    if 3 == len(alpha.shape) and 1 != alpha.shape[-1]:
        raise ValueError("Expecting `alpha` channels size to be 1.")
    if 2 == len(alpha.shape):
        alpha = alpha[..., None]

    if "uint8" != image.dtype or "uint8" != alpha.dtype:
        raise ValueError("Expecting `image` and `alpha` dtype to be `uint8`.")

    image = image.astype("float32") / 255.0
    alpha = alpha.astype("float32") / 255.0

    if 0.0 == np.sum(alpha):
        fg = np.zeros((1, 1, 3))
    else:
        fg = np.sum(image * alpha, axis=(0, 1), keepdims=True) / np.sum(alpha)

    if 0.0 == np.sum(1.0 - alpha):
        bg = np.zeros((1, 1, 3))
    else:
        bg = np.sum(image * (1.0 - alpha), axis=(0, 1), keepdims=True) / np.sum(
            1.0 - alpha
        )

    height, width = image.shape[:2]
    levels = int(np.ceil(np.log2(max(width, height))))

    for level in range(levels + 1):
        height_ = round(height ** (level / levels))
        width_ = round(width ** (level / levels))

        image_ = cv2.resize(
            image, (width_, height_), interpolation=cv2.INTER_LINEAR
        )
        alpha_ = cv2.resize(
            alpha, (width_, height_), interpolation=cv2.INTER_LINEAR
        )
        fg = cv2.resize(fg, (width_, height_), interpolation=cv2.INTER_LINEAR)
        bg = cv2.resize(bg, (width_, height_), interpolation=cv2.INTER_LINEAR)

        steps = (
            small_steps
            if width_ <= small_size and height_ <= small_size
            else big_steps
        )

        for step in range(steps):
            a0 = alpha_[..., None]
            a1 = 1.0 - a0

            a00 = a0**2
            a01 = a0 * a1
            a11 = a1**2

            b0 = a0 * image_
            b1 = a1 * image_

            pad_a0 = np.pad(a0, ((1, 1), (1, 1), (0, 0)), "symmetric")
            da0 = np.abs(a0 - pad_a0[1:-1, :-2]) * grad_weight + regularization
            da1 = np.abs(a0 - pad_a0[1:-1, 2:]) * grad_weight + regularization
            da2 = np.abs(a0 - pad_a0[:-2, 1:-1]) * grad_weight + regularization
            da3 = np.abs(a0 - pad_a0[2:, 1:-1]) * grad_weight + regularization

            da = da0 + da1 + da2 + da3
            a00 += da
            a11 += da

            pad_fg = np.pad(fg, ((1, 1), (1, 1), (0, 0)), "symmetric")
            pad_bg = np.pad(bg, ((1, 1), (1, 1), (0, 0)), "symmetric")

            b0 += (
                da0 * pad_fg[1:-1, :-2]
                + da1 * pad_fg[1:-1, 2:]
                + da2 * pad_fg[:-2, 1:-1]
                + da3 * pad_fg[2:, 1:-1]
            )
            b1 += (
                da0 * pad_bg[1:-1, :-2]
                + da1 * pad_bg[1:-1, 2:]
                + da2 * pad_bg[:-2, 1:-1]
                + da3 * pad_bg[2:, 1:-1]
            )

            inv_det = 1.0 / (a00 * a11 - a01**2)
            b00 = a11 * inv_det
            b01 = -a01 * inv_det
            b11 = a00 * inv_det

            fg = np.clip(b00 * b0 + b01 * b1, 0.0, 1.0)
            bg = np.clip(b01 * b0 + b11 * b1, 0.0, 1.0)

    fg = np.round(fg * 255.0).astype("uint8")
    bg = np.round(bg * 255.0).astype("uint8")

    return fg, bg

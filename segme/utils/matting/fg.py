import cv2
import numpy as np


def solve_fg(image, alpha, kappa=1.0, steps=16):
    # Based on https://github.com/kfeng123/LSA-Matting/blob/master/tools/
    # reestimate_foreground_final.py

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

    height, width = alpha.shape[:2]

    image = image.astype("float32") / 255.0
    alpha = alpha.astype("float32") / 255.0

    afg = alpha * image

    fg = cv2.resize(image, (2, 2), interpolation=cv2.INTER_AREA)
    fg = np.clip(fg, 0.0, 1.0)

    levels = int(np.ceil(np.log2(max(width, height))))
    for level in range(1, levels + 1):
        height_ = round(height ** (level / levels))
        width_ = round(width ** (level / levels))

        afg_ = cv2.resize(afg, (width_, height_), interpolation=cv2.INTER_AREA)
        afg_ = np.clip(afg_, 0.0, 1.0)

        alpha_ = cv2.resize(
            alpha, (width_, height_), interpolation=cv2.INTER_AREA
        )[..., None]
        alpha_ = np.clip(alpha_, 0.0, 1.0)

        fg = cv2.resize(fg, (width_, height_), interpolation=cv2.INTER_LANCZOS4)
        fg = np.clip(fg, 0.0, 1.0)

        a00 = (1 - alpha_) ** 2

        alpha_pad = np.pad(alpha_, ((1, 1), (1, 1), (0, 0)), "reflect")
        da0 = kappa * (a00 + (1 - alpha_pad[1:-1, :-2]) ** 2)
        da1 = kappa * (a00 + (1 - alpha_pad[1:-1, 2:]) ** 2)
        da2 = kappa * (a00 + (1 - alpha_pad[:-2, 1:-1]) ** 2)
        da3 = kappa * (a00 + (1 - alpha_pad[2:, 1:-1]) ** 2)

        denom = alpha_**2 + da0 + da1 + da2 + da3
        term0 = alpha_ * afg_ / denom

        for _ in range(2 ** min(levels - level, 5) * steps):
            fg_pad = np.pad(fg, ((1, 1), (1, 1), (0, 0)), "reflect")
            term1 = (
                da0 * fg_pad[1:-1, :-2]
                + da1 * fg_pad[1:-1, 2:]
                + da2 * fg_pad[:-2, 1:-1]
                + da3 * fg_pad[2:, 1:-1]
            ) / denom
            fg = term0 + term1

    fg = np.clip(fg, 0.0, 1.0)
    fg = np.round(fg * 255.0).astype("uint8")

    return fg

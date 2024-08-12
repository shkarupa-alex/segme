import os

import cv2
import numpy as np
import tensorflow as tf

from segme.utils.matting.np.fgbg import solve_fgbg as solve_fgbg_np
from segme.utils.matting.tf.fgbg import solve_fgbg


class TestSolveFgBg(tf.test.TestCase):
    def test_value(self):
        image = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "np",
            "tests",
            "data",
            "lemur_image.png",
        )
        image = cv2.imread(image)

        alpha = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "np",
            "tests",
            "data",
            "lemur_alpha.png",
        )
        alpha = cv2.imread(alpha, cv2.IMREAD_GRAYSCALE)

        expected_fg, expected_bg = solve_fgbg_np(image, alpha)

        result = solve_fgbg(image[None, ...], alpha[None, ..., None])
        result_fg, result_bg = self.evaluate(result)

        error_fg = np.abs(expected_fg - result_fg[0]).mean()
        error_bg = np.abs(expected_bg - result_bg[0]).mean()

        self.assertLess(error_fg, 0.0018)
        self.assertLess(error_bg, 0.004)

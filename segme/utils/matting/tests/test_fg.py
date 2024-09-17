import os

import cv2
import numpy as np
from keras.src import testing

from segme.utils.matting.fg import solve_fg
from segme.utils.matting_np.fg import solve_fg as solve_fg_np


class TestSolveFg(testing.TestCase):
    def test_value(self):
        image = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "matting_np",
            "tests",
            "data",
            "lemur_image.png",
        )
        image = cv2.imread(image)

        alpha = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "matting_np",
            "tests",
            "data",
            "lemur_alpha.png",
        )
        alpha = cv2.imread(alpha, cv2.IMREAD_GRAYSCALE)

        expected_fg = solve_fg_np(image, alpha)

        result_fg = solve_fg(image[None, ...], alpha[None, ..., None])

        error_fg = np.abs(expected_fg - result_fg[0]).mean()

        self.assertLess(error_fg, 0.11)

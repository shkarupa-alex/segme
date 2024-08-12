import os
import unittest

import cv2
import numpy as np

from segme.utils.matting.np.fgbg import solve_fgbg


class TestSolveFgBg(unittest.TestCase):
    def test_value(self):
        image = os.path.join(
            os.path.dirname(__file__), "data", "lemur_image.png"
        )
        image = cv2.imread(image)

        alpha = os.path.join(
            os.path.dirname(__file__), "data", "lemur_alpha.png"
        )
        alpha = cv2.imread(alpha, cv2.IMREAD_GRAYSCALE)

        trimap = os.path.join(
            os.path.dirname(__file__), "data", "lemur_trimap.png"
        )
        trimap = cv2.imread(trimap, cv2.IMREAD_GRAYSCALE)
        unknown = (trimap > 26) & (trimap < 229)

        foreground = os.path.join(
            os.path.dirname(__file__), "data", "lemur_foreground.png"
        )
        foreground = cv2.imread(foreground)

        fg, _ = solve_fgbg(image, alpha)

        difference = np.abs(foreground - fg) * alpha[..., None]
        error = np.mean(difference[unknown])

        self.assertLess(error, 53.0)


if __name__ == "__main__":
    unittest.main()

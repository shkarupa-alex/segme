import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from segme.utils.matting.tf.fg import solve_fg
from segme.utils.matting.np.fg import solve_fg as solve_fg_np


@test_util.run_all_in_graph_and_eager_modes
class TestSolveFg(tf.test.TestCase):
    def test_value(self):
        image = os.path.join(os.path.dirname(__file__), '..', '..', 'np', 'tests', 'data', 'lemur_image.png')
        image = cv2.imread(image)

        alpha = os.path.join(os.path.dirname(__file__), '..', '..', 'np', 'tests', 'data', 'lemur_alpha.png')
        alpha = cv2.imread(alpha, cv2.IMREAD_GRAYSCALE)

        expected_fg = solve_fg_np(image, alpha)

        result = solve_fg(image[None, ...], alpha[None, ..., None])
        result_fg = self.evaluate(result)

        error_fg = np.abs(expected_fg - result_fg[0]).mean()

        self.assertLess(error_fg, 0.11)


if __name__ == '__main__':
    tf.test.main()

import cv2
import numpy as np
import os
import unittest
from ..compose import compose_two


class TestComposeTwo(unittest.TestCase):
    def test_value(self):
        fg = np.random.uniform(0., 255., (16, 16, 3)).astype('uint8')
        alpha = np.random.uniform(0., 255., (16, 16, 1)).astype('uint8')
        fg_ = np.random.uniform(0., 255., (16, 16, 3)).astype('uint8')
        alpha_ = np.random.uniform(0., 255., (16, 16, 1)).astype('uint8')

        fg__, alpha__ = compose_two(fg, alpha, fg_, alpha_)


if __name__ == '__main__':
    tf.test.main()

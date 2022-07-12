import numpy as np
import unittest
from ..compose import compose_two


class TestComposeTwo(unittest.TestCase):
    def test_compose(self):
        fg0 = np.random.uniform(0., 255., (16, 24, 3)).astype('uint8')
        alpha0 = np.random.uniform(0., 255., (16, 24, 1)).astype('uint8')
        fg1 = np.random.uniform(0., 255., (24, 16, 3)).astype('uint8')
        alpha1 = np.random.uniform(0., 255., (24, 16, 1)).astype('uint8')

        fg, alpha = compose_two(fg0, alpha0, fg1, alpha1, solve=False)

        self.assertEqual(fg.dtype, 'uint8')
        self.assertEqual(alpha.dtype, 'uint8')
        self.assertTupleEqual(fg.shape, fg0.shape)
        self.assertTupleEqual(alpha.shape, alpha0.shape)

    def test_compose_crop(self):
        fg0 = np.random.uniform(0., 255., (16, 24, 3)).astype('uint8')
        alpha0 = np.random.uniform(0., 255., (16, 24, 1)).astype('uint8')
        fg1 = np.random.uniform(0., 255., (28, 24, 3)).astype('uint8')
        alpha1 = np.random.uniform(0., 255., (24, 16, 1)).astype('uint8')

        alpha1 = np.pad(alpha1, [(2, 2), (4, 4), (0, 0)])
        fg, alpha = compose_two(fg0, alpha0, fg1, alpha1, crop=True)

        self.assertEqual(fg.dtype, 'uint8')
        self.assertEqual(alpha.dtype, 'uint8')
        self.assertTupleEqual(fg.shape, fg0.shape)
        self.assertTupleEqual(alpha.shape, alpha0.shape)

    def test_compose_solve(self):
        fg0 = np.random.uniform(0., 255., (16, 24, 3)).astype('uint8')
        alpha0 = np.random.uniform(0., 255., (16, 24, 1)).astype('uint8')
        fg1 = np.random.uniform(0., 255., (24, 16, 3)).astype('uint8')
        alpha1 = np.random.uniform(0., 255., (24, 16, 1)).astype('uint8')

        fg, alpha = compose_two(fg0, alpha0, fg1, alpha1)

        self.assertEqual(fg.dtype, 'uint8')
        self.assertEqual(alpha.dtype, 'uint8')
        self.assertTupleEqual(fg.shape, fg0.shape)
        self.assertTupleEqual(alpha.shape, alpha0.shape)


if __name__ == '__main__':
    unittest.main()

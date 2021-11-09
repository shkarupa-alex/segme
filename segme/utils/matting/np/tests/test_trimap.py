import numpy as np
import unittest
from ..trimap import alpha_trimap


class TestAlphaTrimap(unittest.TestCase):
    def test_shape(self):
        alpha = np.random.randint(0, 255, size=(16, 8), dtype='uint8')

        trimap = alpha_trimap(alpha, 2)
        self.assertTupleEqual((16, 8), trimap.shape)

        trimap = alpha_trimap(alpha[..., None], 2)
        self.assertTupleEqual((16, 8, 1), trimap.shape)

    def test_value(self):
        scale = np.array([0, 1, 1, 3, 7, 15, 31, 63, 127, 255, 255, 255, 255, 255, 255])[..., None]
        alpha = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], 'uint8')
        alpha = np.pad(alpha, 4, mode='edge')
        alpha = alpha * scale.astype('uint8')

        expected = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128],
            [0, 0, 0, 0, 0, 0, 128, 128, 128, 0, 0, 128, 128, 128, 128, 128, 128, 128],
            [0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 0, 128, 128, 128, 128, 128, 128, 128],
            [0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [0, 0, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [0, 0, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [0, 0, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            [0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 0, 128, 128, 128, 128, 128, 128, 128],
            [0, 0, 0, 0, 0, 0, 128, 128, 128, 0, 0, 128, 128, 128, 128, 255, 255, 255],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 255, 255, 255],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 255, 255, 255],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 255, 255, 255]]

        result = alpha_trimap(alpha, 2)
        self.assertListEqual(expected, result.tolist())

    def test_random(self):
        alpha = np.random.randint(0, 255, size=(16, 8), dtype='uint8')
        trimap = alpha_trimap(alpha, (2, 5))

        self.assertEqual(trimap.dtype, 'uint8')
        self.assertTupleEqual(trimap.shape, alpha.shape)


if __name__ == '__main__':
    unittest.main()

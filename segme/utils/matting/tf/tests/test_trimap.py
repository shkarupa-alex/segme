import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from segme.utils.matting.tf.trimap import alpha_trimap
from segme.utils.matting.np.trimap import alpha_trimap as alpha_trimap_np


@test_util.run_all_in_graph_and_eager_modes
class TestAlphaTrimap(tf.test.TestCase):
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

        for size in range(1, 16):
            expected = alpha_trimap_np(alpha, size)
            result = alpha_trimap(alpha[None, ..., None], size)[0, ..., 0]
            result = self.evaluate(result)

            self.assertDTypeEqual(result, 'uint8')
            self.assertListEqual(expected.tolist(), result.tolist())

    def test_random(self):
        alpha = np.random.randint(0, 255, size=(16, 8), dtype='uint8')
        result = alpha_trimap(alpha[None, ..., None], (2, 5))
        result = self.evaluate(result)

        self.assertDTypeEqual(result, 'uint8')


if __name__ == '__main__':
    tf.test.main()

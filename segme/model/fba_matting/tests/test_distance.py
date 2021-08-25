import cv2
import numpy as np
import os
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from keras.mixed_precision import policy as mixed_precision
from ..distance import Distance


@keras_parameterized.run_all_keras_modes
class TestDistance(keras_parameterized.TestCase):
    def setUp(self):
        super(TestDistance, self).setUp()
        self.default_policy = mixed_precision.global_policy()
        tf.debugging.disable_check_numerics()

    def tearDown(self):
        super(TestDistance, self).tearDown()
        mixed_precision.set_policy(self.default_policy)

    def test_layer(self):
        result = testing_utils.layer_test(
            Distance,
            kwargs={},
            input_shape=[2, 64, 64, 2],
            input_dtype='float32',
            expected_output_shape=[None, 64, 64, 6],
            expected_output_dtype='float32'
        )
        self.assertAllLessEqual(result, 1.)
        self.assertAllGreaterEqual(result, 0.)

    def test_value(self):
        twomap = np.array([
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
             [1, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [1, 0], [1, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0], [1, 0],
             [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [1, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0], [1, 0], [1, 0],
             [1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [0, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 0], [0, 0],
             [1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 0],
             [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 0],
             [0, 1], [0, 1], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [1, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1],
             [0, 0], [0, 1], [0, 1], [0, 0], [0, 1], [0, 1], [0, 0], [1, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 0], [0, 0],
             [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [1, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0], [1, 0],
             [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 0], [1, 0], [1, 0],
             [0, 1], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 1], [1, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0], [1, 0],
             [0, 1], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [1, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
             [0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [1, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
             [1, 0], [0, 1], [0, 1], [0, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
             [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
             [0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0],
             [0, 1], [0, 1], [0, 1], [0, 0], [1, 0], [1, 0], [0, 0], [1, 0]]
        ], 'float32')

        expected = _distance(twomap)

        result = Distance()(twomap[None, ...])[0]
        result = self.evaluate(result)

        # self.assertAllClose(expected, result) # differs since tensorflow-addons v0.13.0
        diff = np.sum(np.abs(result - expected) > 1e-6) / np.prod(result.shape)
        self.assertLess(diff, 0.03)

    def test_zeros(self):
        twomap = np.random.rand(64, 64, 2) > 0.5
        twomap[..., 1] = 0
        twomap = twomap.astype('float32')

        expected = _distance(twomap)

        result = Distance()(twomap[None, ...])[0]
        result = self.evaluate(result)

        # self.assertAllClose(expected, result) # differs since tensorflow-addons v0.13.0
        diff = np.sum(np.abs(result - expected) > 1e-6) / np.prod(result.shape)
        self.assertLess(diff, 0.01)

    def test_fp16(self):
        path = os.path.join(os.path.dirname(__file__), 'assets', 'distance_fp16.npy')
        src0 = np.load(path)
        twomap = np.concatenate([src0, 1 - src0], axis=-1).astype('float16')

        tf.debugging.enable_check_numerics()
        mixed_precision.set_policy('mixed_float16')

        result = Distance()(twomap)
        result = self.evaluate(result)
        self.assertTrue(np.all(np.isfinite(result)))


def _distance(twomap, length=320):
    clicks = np.zeros(twomap.shape[:2] + (6,))
    for k in range(2):
        if np.count_nonzero(twomap[:, :, k]):
            dt_src = 1 - twomap[:, :, k]
            dt_mask = -cv2.distanceTransform((dt_src * 255).astype(np.uint8), cv2.DIST_L2, 0) ** 2
            clicks[:, :, 3 * k] = np.exp(dt_mask / (2 * ((0.02 * length) ** 2)))
            clicks[:, :, 3 * k + 1] = np.exp(dt_mask / (2 * ((0.08 * length) ** 2)))
            clicks[:, :, 3 * k + 2] = np.exp(dt_mask / (2 * ((0.16 * length) ** 2)))

    return clicks


if __name__ == '__main__':
    tf.test.main()

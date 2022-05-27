import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.testing_infra import test_combinations, test_utils
from keras.utils.losses_utils import ReductionV2 as Reduction
from ..laplacian_pyramid import LaplacianPyramidLoss
from ..laplacian_pyramid import _gauss_kernel, laplacian_pyramid_loss


@test_combinations.run_all_keras_modes
class TestGaussKernel(test_combinations.TestCase):
    def test_value(self):
        # Differs from https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270#file-laplacian_loss-py-L7
        # expected = np.array(
        #     [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], 'uint8')
        expected = np.array(
            [[1, 4, 6, 4, 1], [4, 15, 24, 15, 4], [6, 24, 37, 24, 6], [4, 15, 24, 15, 4], [1, 4, 6, 4, 1]], 'uint8')

        result = _gauss_kernel(5, 1.06)
        result = result * result.T
        result = np.round(result * 255).astype('uint8')

        self.assertTrue(np.all(expected == result))


@test_combinations.run_all_keras_modes
class TestLaplacianPyramidLoss(test_combinations.TestCase):
    def test_config(self):
        loss = LaplacianPyramidLoss(
            reduction=Reduction.NONE,
            name='loss1'
        )
        self.assertEqual(loss.name, 'loss1')
        self.assertEqual(loss.reduction, Reduction.NONE)

    def test_zeros(self):
        probs = tf.zeros((1, 16, 16, 1), 'float32')
        targets = tf.zeros((1, 16, 16, 1), 'int32')

        result = laplacian_pyramid_loss(y_true=targets, y_pred=probs, sample_weight=None, levels=2, size=5, sigma=2.0)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.], atol=1e-4)

    def test_ones(self):
        probs = tf.ones((1, 16, 16, 1), 'float32')
        targets = tf.ones((1, 16, 16, 1), 'int32')

        result = laplacian_pyramid_loss(y_true=targets, y_pred=probs, sample_weight=None, levels=2, size=5, sigma=2.0)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.], atol=1e-4)

    def test_false(self):
        probs = tf.zeros((1, 16, 16, 1), 'float32')
        targets = tf.ones((1, 16, 16, 1), 'int32')

        result = laplacian_pyramid_loss(y_true=targets, y_pred=probs, sample_weight=None, levels=2, size=5, sigma=2.0)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.35693446], atol=1e-4)

    def test_true(self):
        probs = tf.ones((1, 16, 16, 1), 'float32')
        targets = tf.zeros((1, 16, 16, 1), 'int32')

        result = laplacian_pyramid_loss(y_true=targets, y_pred=probs, sample_weight=None, levels=2, size=5, sigma=2.0)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.35693446], atol=1e-4)

    def test_even(self):
        probs = tf.constant([[
            [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
        ]], 'float32')
        targets = tf.constant([[
            [[0], [0], [0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0]],
        ]], 'int32')

        result = laplacian_pyramid_loss(y_true=targets, y_pred=probs, sample_weight=None, levels=1, size=5, sigma=2.0)
        result = self.evaluate(result)

        self.assertAllClose(result, [0.0])

    def test_multi(self):
        logits = tf.constant([
            [[[0.42, 7.21, 7.14], [7.21, 7.14, 2.55], [7.14, 2.55, 1.34], [2.55, 1.34, 0.20]],
             [[1.34, 0.20, 3.97], [0.20, 3.97, 6.28], [3.97, 6.28, 0.32], [6.28, 0.32, 3.01]],
             [[0.32, 3.01, 2.90], [3.01, 2.90, 3.36], [2.90, 3.36, 2.65], [3.36, 2.65, 6.86]],
             [[2.65, 6.86, 4.58], [6.86, 4.58, 7.43], [4.58, 7.43, 8.13], [7.43, 8.13, 8.31]]],
            [[[8.13, 8.31, 0.83], [8.31, 0.83, 2.85], [0.83, 2.85, 2.09], [2.85, 2.09, 4.61]],
             [[2.09, 4.61, 8.70], [4.61, 8.70, 1.91], [8.70, 1.91, 3.49], [1.91, 3.49, 4.55]],
             [[3.49, 4.55, 7.70], [4.55, 7.70, 3.39], [7.70, 3.39, 0.91], [3.39, 0.91, 3.03]],
             [[0.91, 3.03, 2.18], [3.03, 2.18, 1.39], [2.18, 1.39, 0.42], [1.39, 0.42, 7.21]]]], 'float32')
        targets = tf.constant([
            [[[0, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]], [[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1]],
             [[0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1]], [[0, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 1]]],
            [[[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0]], [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1]],
             [[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]]], 'float32')
        weights = tf.concat([tf.ones((2, 4, 2, 1)), tf.zeros((2, 4, 2, 1))], axis=2)

        loss = LaplacianPyramidLoss(reduction=Reduction.SUM, levels=1)

        result = self.evaluate(loss(targets, logits, weights))
        self.assertAlmostEqual(result, 2.0541933, places=6)

    def test_value(self):
        probs = tf.constant([
            0.5, 6.1, 7.2, 9.0, 7.3, 1.7, 3.1, 7.8, 7.7, 9.8, 0.7, 0.6, 7.4, 9.9, 4.7, 1.5, 7.9, 5.4, 9.2, 9.1, 9.9,
            4.3, 9.3, 1.1, 8.6, 3.8, 6.6, 9.4, 8.9, 7.8, 4.3, 5.1, 8.1, 9.9, 2.4, 3.0, 0.9, 9.6, 0.1, 4.2, 8.3, 6.6,
            2.8, 5.8, 1.1, 0.6, 2.9, 2.8, 4.9, 0.5, 8.3, 2.3, 1.2, 2.4, 8.3, 0.1, 5.4, 4.8, 1.7, 8.2, 9.7, 2.3, 7.8,
            7.7, 6.3, 4.8, 1.7, 8.4, 2.5, 6.0, 0.5, 7.4, 2.2, 9.8, 5.8, 0.5, 4.6, 1.4, 2.1, 1.5, 1.1, 6.7, 0.7, 4.8,
            5.5, 5.5, 4.1, 8.2, 2.4, 1.8, 8.3, 6.1, 0.6, 7.5, 4.8, 6.2, 0.2, 9.6, 9.9, 6.8, 7.3, 7.9, 8.8, 7.9, 7.6,
            2.3, 7.0, 5.8, 9.5, 1.9, 6.4, 1.5, 0.1, 9.3, 6.9, 4.4, 5.7, 5.8, 0.9, 9.0, 1.3, 4.5, 9.8, 5.0, 4.6, 7.8,
            8.6, 3.8, 1.3, 3.2, 0.8, 7.6, 4.9, 3.5, 2.7, 6.1, 1.4, 7.9, 0.4, 0.4, 8.0, 2.0, 0.3, 7.6, 4.0, 0.0, 5.9,
            2.4, 2.0, 7.7, 7.0, 2.5, 3.5, 7.4, 2.4, 4.7, 3.0, 4.4, 2.5, 0.0, 3.5, 0.6, 3.5, 4.2, 7.2, 7.0, 2.4, 7.9,
            9.8, 4.7, 6.1, 9.8, 8.2, 4.0, 1.8, 4.8, 6.4, 9.3, 8.8, 5.7, 3.9, 3.5, 4.4, 7.1, 3.9, 5.5, 2.5, 9.1, 6.7,
            4.1, 5.2, 4.9, 2.6, 4.0, 2.1, 4.6, 5.7, 9.0, 6.5, 4.4, 6.4, 2.9, 9.3, 6.2, 0.0, 7.4, 0.8, 8.3, 3.0, 6.2,
            0.9, 0.3, 3.7, 2.7, 8.4, 0.8, 3.8, 9.3, 4.7, 2.6, 1.3, 2.3, 6.4, 8.2, 7.3, 2.7, 4.8, 7.8, 4.9, 0.4, 4.0,
            3.1, 6.4, 3.7, 6.8, 0.1, 9.8, 6.2, 1.7, 6.6, 2.8, 9.3, 4.1, 9.6, 6.0, 5.7, 3.3, 3.4, 0.7, 7.5, 1.4, 3.2,
            5.5, 5.1, 4.6, 1.6, 0.9, 4.7, 1.5, 8.8, 6.1, 6.9, 7.8, 6.6, 5.5, 5.6, 3.5, 9.9, 9.1, 5.8, 5.3, 3.5, 1.5,
            9.1, 6.0, 6.8, 7.1, 7.6, 6.2, 9.9, 9.9, 0.5, 8.3, 9.2, 1.5, 4.6, 7.1, 3.5, 3.7, 6.3, 2.3, 3.8, 6.1, 1.7,
            3.2, 4.3, 6.1, 7.9, 7.9, 3.4, 6.6, 0.6, 5.2, 1.7, 2.4, 1.4, 0.8, 9.9, 2.5, 2.6, 0.2, 1.5, 2.2, 9.0, 7.3,
            1.3, 8.0, 5.9, 6.2, 4.8, 7.9, 8.8, 3.8, 4.3, 4.3, 1.9, 8.2, 5.7, 2.5, 1.6, 5.2, 3.6, 3.3, 5.1, 5.8, 8.5,
            0.2, 0.0, 0.7, 1.4, 0.1, 4.6, 8.7, 3.0, 3.1, 5.2, 1.7, 0.0, 4.4, 7.8, 9.7, 4.1, 3.9, 1.3, 8.0, 2.8, 6.4,
            5.5, 2.0, 9.7, 1.8, 9.1, 9.4, 1.9, 0.4, 0.3, 5.2, 2.4, 0.2, 3.3, 0.8, 1.2, 2.2, 1.5, 5.5, 3.3, 0.3, 1.9,
            0.6, 1.2, 1.1, 1.5, 4.5, 2.7, 4.3, 1.5, 5.4, 9.1, 1.7, 9.0, 4.8, 3.9, 3.5, 5.5, 7.0, 1.5, 4.9, 4.6, 9.6,
            5.9, 9.3, 8.5, 5.5, 7.4, 1.1, 6.7, 2.4, 8.4, 0.7, 8.5, 3.0, 5.3, 5.0, 8.6, 6.2, 0.7, 6.6, 4.1, 8.9, 2.1,
            1.4, 6.9, 5.3, 2.6, 2.8, 1.0, 0.8, 5.4, 0.3, 7.7, 8.2, 9.2, 7.2, 5.0, 0.3, 3.5, 6.6, 6.2, 0.3, 3.2, 0.2,
            3.5, 5.0, 4.8, 0.7, 0.7, 3.0, 9.0, 9.5, 0.5, 2.7, 3.5, 2.7, 7.9, 1.2, 5.1, 5.6, 6.5, 2.2, 2.6, 6.8, 4.7,
            2.9, 0.7, 6.9, 7.6, 6.0, 0.1, 6.8, 2.3, 9.1, 5.8, 6.5, 0.9, 9.4, 3.3, 8.4, 9.9, 4.5, 4.4, 2.0, 6.8, 3.4,
            8.1, 6.5, 0.4, 3.1, 3.9, 3.6, 4.9, 8.9, 4.1, 2.6, 4.7, 0.4, 3.2, 1.2, 4.3, 2.8, 6.1, 6.0, 6.0, 1.7, 7.6,
            7.2, 2.8, 3.0, 1.9, 7.0, 0.7, 0.7, 1.6, 7.4, 0.8, 4.2, 7.3, 5.5, 3.9, 3.9, 4.6, 3.0, 5.0, 5.4, 5.7, 5.5,
            1.1, 7.6, 9.3, 1.8, 9.6, 9.6, 9.1, 7.5, 3.0, 7.7, 3.7, 7.4, 3.8, 6.6, 8.6, 9.3, 8.0, 7.8, 8.6, 9.1, 8.4,
            5.2, 0.9, 5.2, 9.6, 5.1, 6.9, 2.5, 4.8, 4.6, 6.6, 6.9, 7.3, 7.6, 2.5, 2.9, 0.8, 8.5, 6.0, 4.5, 3.4, 4.8,
            6.7, 7.8, 9.7, 1.2, 5.0, 6.4, 3.7, 3.4, 4.1, 0.5, 7.9, 9.3, 8.6, 0.8, 8.0, 3.6, 6.4, 3.1, 8.4, 3.8, 2.9,
            5.5, 1.8, 8.9, 4.3, 2.7, 4.5, 3.0, 6.3, 0.9, 8.0, 7.8, 2.9, 0.0, 7.4, 6.3, 9.4, 9.5, 2.4, 2.3, 3.4, 0.5,
            4.3, 8.3, 9.1, 0.5, 0.1, 7.5, 2.3, 4.6, 9.6, 7.2, 9.4, 2.8, 6.3, 1.3, 2.2, 9.7, 3.6, 8.0, 4.9, 0.0, 7.4,
            9.6, 0.2, 1.2, 3.5, 1.8, 9.3, 2.9, 7.4, 2.7, 9.5, 4.5, 7.1, 4.4, 4.4, 4.2, 1.6, 3.0, 7.8, 2.1, 6.2, 8.7,
            0.8, 6.3, 0.5, 6.8, 3.0, 7.9, 5.0, 7.7, 4.2, 6.5, 8.7, 5.3, 2.6, 6.7, 9.5, 9.4, 7.3, 8.2, 7.4, 6.8, 9.3,
            9.0, 2.2, 7.3, 9.3, 0.2, 9.6, 6.2, 0.1, 1.9, 4.7, 3.6, 3.9, 7.8, 4.8, 4.1, 5.8, 8.1, 2.2, 8.6, 3.8, 7.3,
            9.6, 1.4, 3.3, 0.5, 3.4, 7.4, 4.0, 0.4, 8.7, 2.6, 1.9, 5.1, 7.1, 5.8, 2.0, 2.6, 4.7, 9.9, 0.1, 8.0, 4.9,
            6.7, 2.9, 3.0, 2.3, 9.6, 6.5, 8.0, 5.8, 9.0, 6.9, 5.1, 7.0, 4.8, 1.4, 2.0, 1.1, 4.0, 2.6, 0.4, 0.4, 1.9,
            2.6, 7.5, 0.7, 6.5, 6.2, 6.9, 4.0, 6.6, 5.4, 9.1, 2.7, 4.2, 3.3, 1.9, 2.1, 3.4, 7.2, 9.7, 3.1, 3.7, 3.0,
            6.4, 1.9, 1.6, 8.2, 7.8, 8.2, 2.4, 1.0, 9.2, 7.1, 7.2, 0.1, 0.7, 0.5, 9.2, 1.3, 5.0, 4.0, 1.0, 7.4, 9.9,
            2.2, 2.8, 7.7, 7.1, 5.7, 7.2, 9.2, 3.0, 8.3, 8.4, 1.7, 3.4, 8.0, 7.6, 1.5, 2.8, 6.4, 0.5, 8.2, 0.7, 0.5,
            2.1, 0.1, 3.4, 7.2, 9.5, 6.2, 9.0, 1.2, 0.8, 8.0, 1.3, 4.1, 9.9, 8.2, 5.3, 1.6, 0.2, 3.4, 3.8, 2.7, 9.3,
            0.9, 4.8, 6.9, 7.4, 4.5, 2.8, 4.3, 5.2, 3.6, 9.7, 8.1, 3.4, 4.0, 8.9, 3.1, 4.1, 1.5, 7.1, 5.9, 7.6, 4.4,
            5.5, 0.6, 3.0, 9.9, 3.3, 7.9, 1.4, 7.8, 7.3, 9.2, 1.9, 4.0, 7.1, 7.3, 3.0, 0.1, 9.6, 4.1, 3.3, 3.8, 4.6,
            3.3, 3.6, 5.5, 2.6, 1.5, 0.0, 7.4, 6.6, 6.0, 8.7, 5.4, 1.1, 6.5, 3.9, 9.4, 6.2, 4.6, 4.9, 3.5, 6.0, 6.3,
            5.5, 0.6, 4.8, 1.0, 3.8, 7.3, 0.1, 4.5, 8.8, 5.1, 5.4, 7.4, 7.4, 7.3, 2.7, 6.2, 1.7, 9.9, 0.1, 4.6, 3.8,
            6.2, 7.3, 8.4, 9.8, 9.2, 5.8, 9.8, 8.7, 1.1, 9.3, 9.2, 0.8, 7.8, 0.7, 0.0, 3.0, 7.7, 4.3, 3.4, 1.6, 6.4,
            2.5, 8.5, 2.1, 8.3, 1.3, 1.3, 6.0, 5.1, 2.1, 6.2, 1.3, 1.2, 7.7, 3.0, 0.3, 3.5, 3.9, 1.8, 0.8, 5.1, 4.6,
            4.2, 9.5, 4.4, 5.6, 2.6, 0.0, 0.8, 2.4, 5.3, 0.0, 4.0, 1.7, 2.0, 2.4, 4.1, 6.7, 4.7, 6.2, 1.2, 8.6, 3.6,
            2.1, 8.6, 3.4, 1.1, 5.4, 1.8, 6.4, 2.9, 2.5, 7.4, 5.2, 1.9, 3.3, 0.1, 7.5, 1.8, 9.4, 2.5, 9.3, 7.7, 5.1,
            3.5, 2.2, 5.5, 9.5, 3.3, 4.6, 7.6, 9.2, 6.0, 7.4, 2.5, 7.4, 8.1, 5.9, 7.2, 6.4, 1.1, 3.9, 9.1, 1.0, 0.8,
            6.3, 2.5, 2.1, 4.3, 1.2, 3.9, 4.0, 1.3, 8.6, 9.7, 3.0, 3.0, 5.3, 2.2, 0.4, 1.0
        ], 'float32', shape=(1, 32, 32, 1))
        targets = tf.constant([
            9.9, 4.6, 2.7, 3.8, 9.0, 5.0, 9.7, 5.9, 6.1, 3.9, 5.9, 2.8, 2.1, 8.5, 7.8, 1.6, 5.6, 3.0, 1.4, 1.1, 7.6,
            5.4, 1.3, 2.4, 9.8, 9.2, 5.6, 5.2, 0.7, 6.0, 7.6, 3.9, 8.4, 5.3, 1.0, 7.2, 6.9, 2.8, 3.8, 9.6, 1.7, 7.5,
            3.1, 1.8, 1.9, 4.2, 8.5, 2.8, 1.5, 0.6, 3.6, 9.4, 4.5, 2.5, 6.6, 1.8, 8.6, 7.6, 3.1, 0.4, 5.1, 6.7, 3.0,
            9.7, 7.7, 3.3, 3.1, 2.3, 2.4, 8.0, 9.8, 9.4, 3.8, 1.0, 0.7, 7.5, 0.7, 3.9, 5.5, 1.6, 1.6, 0.8, 5.5, 2.3,
            5.8, 6.2, 1.6, 4.2, 7.9, 0.4, 7.0, 1.3, 1.2, 5.7, 6.2, 6.4, 2.7, 7.6, 2.7, 9.2, 2.1, 5.2, 9.3, 2.5, 6.1,
            5.7, 3.5, 0.5, 1.3, 1.8, 8.9, 7.5, 7.2, 6.4, 6.9, 6.4, 6.5, 6.3, 9.3, 7.5, 2.6, 3.2, 0.9, 1.0, 9.1, 9.4,
            8.1, 2.6, 8.1, 4.9, 0.8, 5.1, 9.6, 5.1, 8.5, 4.5, 7.4, 9.0, 8.8, 4.1, 8.5, 6.0, 1.5, 2.2, 0.7, 0.6, 8.3,
            8.4, 8.6, 7.8, 9.1, 0.5, 2.5, 1.0, 6.6, 9.8, 0.8, 5.9, 2.7, 9.6, 1.5, 6.4, 9.6, 5.4, 2.1, 3.3, 9.2, 3.0,
            2.5, 8.7, 2.3, 0.4, 6.8, 4.7, 7.4, 1.4, 6.1, 9.0, 9.3, 8.2, 0.7, 5.4, 3.4, 8.1, 7.5, 0.2, 3.1, 7.0, 7.9,
            9.9, 2.1, 5.2, 7.8, 5.8, 2.6, 2.7, 6.2, 3.8, 1.1, 8.1, 3.0, 8.3, 9.0, 2.1, 8.9, 0.3, 0.3, 5.1, 2.2, 3.1,
            9.3, 1.9, 2.2, 9.6, 4.8, 0.6, 3.3, 3.4, 2.0, 8.2, 7.8, 8.2, 5.4, 7.7, 1.1, 9.6, 4.9, 4.8, 0.5, 0.8, 5.2,
            3.7, 5.7, 6.7, 3.5, 9.2, 2.4, 2.6, 0.4, 1.2, 8.5, 0.2, 7.0, 8.3, 2.8, 0.7, 4.3, 7.0, 3.7, 3.8, 8.3, 9.6,
            9.9, 9.1, 4.1, 5.4, 6.7, 3.9, 6.4, 6.0, 1.1, 3.7, 9.3, 0.1, 5.7, 4.6, 2.9, 0.2, 2.8, 2.7, 0.2, 6.4, 9.9,
            5.4, 0.4, 4.7, 5.1, 9.2, 5.9, 4.2, 5.4, 2.2, 3.7, 4.0, 7.1, 1.9, 9.4, 5.9, 8.4, 5.1, 1.8, 6.0, 3.5, 1.8,
            6.7, 9.7, 2.3, 8.4, 1.8, 6.2, 2.6, 7.8, 6.1, 4.6, 7.6, 0.8, 4.5, 7.1, 5.6, 6.9, 6.9, 0.8, 2.3, 4.4, 4.1,
            6.1, 0.1, 5.6, 8.1, 1.3, 2.0, 5.4, 8.0, 8.3, 0.7, 9.6, 5.5, 1.6, 4.2, 5.6, 2.4, 5.9, 9.9, 4.9, 4.2, 0.7,
            7.2, 1.4, 7.3, 2.0, 4.6, 7.5, 6.0, 9.9, 1.5, 0.1, 2.4, 7.8, 2.3, 9.3, 7.5, 7.9, 2.4, 9.6, 5.2, 8.2, 0.4,
            1.6, 0.2, 1.6, 9.0, 7.0, 0.5, 0.9, 8.2, 6.1, 1.1, 1.3, 6.3, 0.6, 6.3, 0.8, 4.0, 4.9, 7.2, 0.1, 1.8, 5.9,
            7.8, 3.5, 9.3, 5.4, 7.7, 3.8, 3.4, 6.1, 0.2, 1.5, 6.2, 5.1, 8.3, 2.9, 1.4, 1.1, 1.9, 2.3, 6.1, 1.4, 4.8,
            1.4, 2.0, 9.2, 2.8, 6.6, 5.3, 1.5, 3.0, 5.0, 9.2, 0.5, 3.6, 2.7, 0.6, 2.4, 3.4, 3.7, 8.7, 8.9, 2.0, 2.8,
            0.9, 7.6, 3.2, 8.1, 2.1, 8.6, 0.4, 8.3, 6.0, 5.0, 4.8, 6.6, 9.1, 7.9, 4.5, 7.7, 7.0, 6.2, 3.8, 5.3, 4.4,
            8.0, 7.9, 3.7, 3.6, 8.3, 0.2, 1.9, 6.1, 9.4, 9.0, 2.9, 9.1, 8.8, 0.2, 3.9, 7.2, 1.3, 0.4, 4.4, 6.7, 1.1,
            7.2, 1.5, 0.3, 5.6, 6.4, 1.2, 2.1, 4.7, 0.1, 9.8, 9.1, 2.8, 7.3, 6.2, 7.6, 4.2, 6.4, 6.2, 3.7, 7.2, 5.9,
            6.2, 7.3, 1.4, 0.1, 5.1, 6.7, 0.0, 4.9, 0.6, 0.1, 7.4, 2.2, 1.9, 2.1, 4.8, 1.9, 6.6, 4.8, 6.4, 7.9, 7.7,
            6.2, 9.7, 8.1, 6.9, 1.7, 4.0, 9.9, 5.7, 0.9, 8.6, 7.2, 7.0, 7.0, 5.7, 2.8, 1.2, 7.4, 7.9, 3.1, 5.1, 9.4,
            6.6, 6.4, 6.3, 2.5, 4.6, 5.0, 7.2, 1.8, 6.1, 8.9, 9.1, 8.6, 3.6, 7.8, 8.0, 5.7, 6.7, 1.1, 2.8, 3.8, 4.9,
            2.1, 5.9, 2.2, 8.2, 3.7, 6.3, 4.0, 3.3, 8.0, 3.3, 5.5, 0.8, 3.4, 4.4, 7.9, 7.6, 0.7, 0.7, 9.2, 6.1, 9.9,
            1.1, 9.8, 2.6, 4.5, 3.8, 5.4, 8.8, 8.6, 3.0, 4.7, 9.3, 2.6, 1.6, 4.1, 7.5, 4.4, 4.2, 7.0, 7.0, 2.0, 1.6,
            3.5, 4.1, 7.2, 9.7, 0.1, 2.2, 0.2, 6.0, 9.4, 0.1, 7.4, 1.9, 8.0, 3.7, 2.8, 2.0, 6.1, 3.2, 6.6, 9.4, 4.0,
            0.9, 9.1, 1.9, 4.9, 8.3, 3.9, 1.6, 6.2, 3.4, 1.9, 2.1, 1.9, 1.5, 2.7, 6.3, 9.9, 0.4, 5.2, 4.9, 3.0, 2.4,
            7.1, 7.3, 2.8, 1.5, 1.7, 7.2, 5.5, 5.5, 8.4, 0.8, 0.4, 9.5, 0.5, 3.6, 0.9, 2.3, 3.3, 2.5, 9.9, 4.6, 7.7,
            4.2, 1.0, 8.7, 6.2, 8.5, 8.8, 2.1, 0.5, 1.3, 1.3, 6.6, 9.0, 7.6, 2.9, 6.6, 0.6, 3.1, 0.1, 9.0, 4.2, 4.8,
            1.9, 2.5, 7.6, 5.1, 6.2, 2.6, 3.9, 3.2, 5.2, 4.3, 0.8, 1.5, 6.3, 7.6, 5.4, 3.3, 3.4, 4.7, 9.8, 5.7, 3.8,
            9.0, 9.5, 3.7, 7.4, 0.3, 3.4, 4.8, 2.1, 1.4, 1.0, 1.2, 8.6, 0.1, 4.5, 6.0, 8.0, 8.9, 7.2, 7.6, 8.6, 5.9,
            3.5, 9.1, 0.2, 8.0, 3.2, 1.5, 9.8, 9.8, 1.5, 5.2, 8.2, 9.6, 6.1, 3.4, 5.3, 7.9, 3.7, 1.8, 6.4, 9.1, 8.6,
            6.5, 9.8, 7.4, 4.1, 0.6, 7.7, 6.5, 7.8, 7.6, 6.8, 3.9, 8.0, 0.6, 8.9, 3.6, 0.6, 9.6, 4.7, 5.7, 1.8, 7.7,
            4.7, 3.2, 1.8, 9.6, 4.6, 8.2, 5.2, 9.8, 8.4, 5.1, 7.7, 5.3, 0.7, 0.3, 0.0, 7.8, 0.5, 2.7, 2.6, 4.6, 3.7,
            3.8, 6.6, 6.9, 7.7, 8.0, 1.7, 5.4, 8.2, 2.4, 7.7, 7.1, 4.1, 8.0, 1.2, 1.4, 3.1, 5.0, 4.7, 9.0, 2.2, 5.4,
            5.2, 2.1, 4.1, 2.3, 5.9, 4.3, 9.7, 5.8, 4.7, 0.6, 7.1, 5.4, 6.8, 4.9, 5.5, 3.5, 3.5, 6.5, 9.5, 3.1, 6.7,
            4.9, 7.5, 4.7, 5.8, 1.5, 1.0, 4.6, 8.0, 3.2, 3.7, 5.3, 6.5, 6.2, 4.8, 9.5, 6.8, 4.0, 6.2, 1.0, 9.1, 5.5,
            3.8, 8.1, 6.3, 8.7, 9.4, 5.7, 3.2, 8.8, 1.7, 1.8, 6.2, 3.7, 1.1, 3.2, 9.1, 1.1, 7.1, 7.0, 0.9, 8.2, 8.1,
            0.3, 7.8, 4.2, 7.7, 8.5, 5.1, 3.1, 8.8, 4.9, 5.2, 2.2, 3.3, 2.4, 1.5, 8.4, 3.2, 7.0, 1.2, 1.8, 8.5, 1.1,
            7.9, 8.0, 2.2, 1.0, 2.7, 5.0, 1.0, 8.8, 5.0, 1.3, 3.5, 9.7, 0.5, 8.7, 2.4, 1.1, 0.1, 4.4, 4.7, 0.8, 5.5,
            5.1, 9.2, 8.2, 3.0, 4.7, 0.8, 8.4, 7.1, 8.8, 2.3, 3.6, 7.8, 3.0, 4.8, 3.0, 1.7, 2.9, 6.8, 5.6, 1.0, 9.6,
            1.5, 5.3, 3.4, 2.0, 1.3, 4.7, 7.0, 0.8, 9.2, 5.6, 6.8, 0.2, 7.9, 3.6, 6.5, 9.4, 1.3, 0.2, 4.2, 3.8, 3.1,
            7.9, 5.5, 7.1, 8.8, 3.2, 8.4, 2.5, 9.3, 5.5, 5.6, 8.1, 8.2, 2.0, 1.5, 4.1, 8.5, 7.5, 4.9, 1.7, 3.3, 0.5,
            9.7, 7.6, 4.5, 8.3, 8.0, 8.9, 7.8, 0.0, 3.1, 0.4, 9.5, 5.9, 9.7, 2.7, 6.0, 7.6, 3.0, 2.5, 5.5, 9.4, 1.2,
            6.2, 6.0, 7.3, 3.6, 2.5, 7.8, 3.5, 2.0, 0.6, 5.3, 6.5, 1.7, 9.1, 6.3, 4.2, 9.0, 5.2, 6.4, 1.8, 1.3, 8.5,
            1.4, 9.2, 5.7, 8.9, 9.5, 0.4, 3.3, 7.7, 0.2, 5.6, 9.5, 6.4, 0.5, 0.0, 3.2, 3.1
        ], 'float32', shape=(1, 32, 32, 1))

        loss = LaplacianPyramidLoss(levels=4, size=5, sigma=1.056, reduction=Reduction.SUM)
        result = self.evaluate(loss(targets, probs))

        # self.assertAlmostEqual(result, 6.944647789001465, places=6) # FBA (without residual)
        # self.assertAlmostEqual(result, 7.640613555908203, places=6)  # with residual
        self.assertAlmostEqual(result, 6.970603942871094, places=5)  # without residual

    def test_weight(self):
        logits = tf.constant([
            [[[0.4250706654827763], [7.219920928747051], [7.14131948950217], [2.5576064452206024]],
             [[1.342442193620409], [0.20020616879804165], [3.977300484664198], [6.280817910206608]],
             [[0.3206719246447576], [3.0176225602425912], [2.902292891065069], [3.369106587128292]],
             [[2.6576544216404563], [6.863726154333165], [4.581314280496405], [7.433728759092233]]],
            [[[8.13888654097292], [8.311411218599392], [0.8372454481780323], [2.859455217953778]],
             [[2.0984725413538854], [4.619268334888168], [8.708732477440673], [1.9102341271004541]],
             [[3.4914178176388266], [4.551627675234152], [7.709902261544302], [3.3982255596983277]],
             [[0.9182162683255968], [3.0387004793287886], [2.1883984916630697], [1.3921544038795197]]]], 'float32')
        targets = tf.constant([
            [[[0], [0], [1], [0]], [[1], [0], [1], [1]], [[0], [1], [0], [1]], [[0], [1], [1], [1]]],
            [[[0], [1], [1], [0]], [[1], [0], [0], [1]], [[0], [1], [1], [0]], [[1], [1], [1], [1]]]], 'int32')
        logits = tf.repeat(tf.repeat(logits, 16, axis=1), 16, axis=2)
        targets = tf.repeat(tf.repeat(targets, 16, axis=1), 16, axis=2)
        weights = tf.concat([tf.ones((2, 64, 32, 1)), tf.zeros((2, 64, 32, 1))], axis=2)

        loss = LaplacianPyramidLoss(reduction=Reduction.SUM, levels=2)

        result = self.evaluate(loss(targets, logits))
        self.assertAlmostEqual(result, 2.27636, places=6)

        result = self.evaluate(loss(targets[:, :, :32, :], logits[:, :, :32, :]))
        self.assertAlmostEqual(result, 1.8797371, places=7)

        result = self.evaluate(loss(targets, logits, weights))
        self.assertAlmostEqual(result, 1.0742356, places=5)

        result = self.evaluate(loss(targets, logits, weights * 2.))
        self.assertAlmostEqual(result, 1.0742356 * 2., places=6)

    def test_batch(self):
        probs = np.random.rand(2, 128, 128, 3).astype('float32')
        targets = (np.random.rand(2, 128, 128, 3) > 0.5).astype('int32')

        loss = LaplacianPyramidLoss(reduction=Reduction.SUM_OVER_BATCH_SIZE)
        res0 = self.evaluate(loss(targets, probs))
        res1 = sum([self.evaluate(loss(targets[i:i + 1], probs[i:i + 1])) for i in range(2)]) / 2

        self.assertAlmostEqual(res0, res1, places=6)

    def test_model(self):
        model = models.Sequential([layers.Dense(3, activation='sigmoid')])
        model.compile(loss='SegMe>LaplacianPyramidLoss', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.zeros((2, 224, 224, 5)), np.zeros((2, 224, 224, 3), 'int32'))
        models.Sequential.from_config(model.get_config())


if __name__ == '__main__':
    tf.test.main()

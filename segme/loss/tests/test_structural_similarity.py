import numpy as np
import tensorflow as tf
from tf_keras import layers, models
from tf_keras.src.testing_infra import test_combinations, test_utils
from tf_keras.src.utils.losses_utils import ReductionV2 as Reduction
from segme.loss.structural_similarity import StructuralSimilarityLoss
from segme.loss.structural_similarity import _ssim_kernel, _ssim_level, structural_similarity_loss


@test_combinations.run_all_keras_modes
class TestSsimLevel(test_combinations.TestCase):
    def test_value(self):
        y_true = np.array([
            0., 0., 0.01, 0.03, 0.09, 0.04, 0.1, 0.18, 0.11, 0.27, 0.2, 0.29, 0.2, 0.36, 0.38, 0.43, 0.47, 0.44, 0.57,
            0.74, 0.78, 0.8, 0.01, 0.02, 0.05, 0.05, 0.14, 0.11, 0.19, 0.23, 0.2, 0.29, 0.39, 0.34, 0.47, 0.42, 0.57,
            0.59, 0.58, 0.67, 0.72, 0.76, 0.84, 0.87, 0.01, 0.03, 0.1, 0.07, 0.15, 0.12, 0.23, 0.27, 0.42, 0.3, 0.44,
            0.36, 0.48, 0.45, 0.62, 0.6, 0.71, 0.68, 0.73, 0.8, 0.87, 0.88, 0.02, 0.03, 0.1, 0.1, 0.15, 0.13, 0.24,
            0.27, 0.44, 0.34, 0.44, 0.39, 0.48, 0.46, 0.62, 0.64, 0.75, 0.7, 0.76, 0.82, 0.89, 0.9, 0.02, 0.04, 0.11,
            0.1, 0.2, 0.13, 0.35, 0.28, 0.45, 0.34, 0.46, 0.42, 0.5, 0.47, 0.65, 0.65, 0.76, 0.73, 0.83, 0.83, 0.94,
            0.9, 0.07, 0.05, 0.12, 0.12, 0.21, 0.24, 0.35, 0.33, 0.45, 0.35, 0.5, 0.47, 0.58, 0.54, 0.68, 0.67, 0.8,
            0.74, 0.85, 0.87, 0.95, 0.95, 0.08, 0.07, 0.14, 0.14, 0.22, 0.24, 0.36, 0.36, 0.47, 0.45, 0.55, 0.48, 0.59,
            0.54, 0.71, 0.7, 0.83, 0.77, 0.86, 0.88, 0.95, 0.95, 0.08, 0.07, 0.19, 0.17, 0.31, 0.26, 0.37, 0.4, 0.51,
            0.47, 0.59, 0.53, 0.65, 0.55, 0.74, 0.73, 0.85, 0.83, 0.91, 0.88, 0.96, 0.97, 0.09, 0.09, 0.21, 0.21, 0.31,
            0.29, 0.41, 0.46, 0.52, 0.54, 0.61, 0.56, 0.66, 0.62, 0.78, 0.74, 0.88, 0.86, 0.96, 0.93, 0.97, 0.98, 0.13,
            0.15, 0.3, 0.23, 0.32, 0.35, 0.42, 0.59, 0.56, 0.63, 0.62, 0.64, 0.66, 0.7, 0.8, 0.75, 0.95, 0.93, 0.96,
            0.95, 0.98, 0.98, 0.22, 0.24, 0.56, 0.34, 0.7, 0.37, 0.76, 0.6, 0.81, 0.69, 0.86, 0.72, 0.88, 0.88, 0.93,
            0.92, 0.95, 0.95, 0.96, 0.96, 0.99, 1.]).reshape((1, 11, 11, 2)).astype('float32')
        y_pred = np.array([
            0.2, -0.2, 0.21, -0.17, 0.29, 0.24, 0.3, -0.02, 0.31, 0.47, -0., 0.09, 0.4, 0.16, 0.18, 0.23, 0.67, 0.64,
            0.77, 0.94, 0.98, 0.6, -0.19, 0.22, 0.25, -0.15, -0.06, -0.09, -0.01, 0.43, -0., 0.49, 0.19, 0.54, 0.67,
            0.62, 0.77, 0.39, 0.38, 0.87, 0.92, 0.96, 1.04, 1.07, 0.21, 0.23, -0.1, -0.13, 0.35, 0.32, 0.03, 0.47, 0.22,
            0.5, 0.24, 0.16, 0.28, 0.65, 0.42, 0.8, 0.91, 0.88, 0.53, 0.6, 0.67, 1.08, 0.22, 0.23, -0.1, 0.3, -0.05,
            0.33, 0.44, 0.47, 0.64, 0.14, 0.25, 0.19, 0.28, 0.26, 0.82, 0.83, 0.95, 0.5, 0.56, 0.62, 0.69, 0.7, -0.18,
            -0.16, -0.09, -0.1, 0., -0.07, 0.55, 0.08, 0.65, 0.15, 0.26, 0.62, 0.3, 0.27, 0.45, 0.85, 0.96, 0.53, 1.02,
            1.03, 0.74, 1.1, 0.27, 0.25, 0.32, -0.08, 0.01, 0.43, 0.15, 0.53, 0.25, 0.15, 0.3, 0.67, 0.38, 0.74, 0.48,
            0.47, 1., 0.54, 0.65, 0.67, 1.15, 1.15, -0.12, 0.27, -0.06, 0.34, 0.42, 0.04, 0.56, 0.16, 0.67, 0.25, 0.75,
            0.68, 0.79, 0.34, 0.51, 0.9, 0.63, 0.97, 0.66, 0.68, 0.75, 1.15, 0.28, -0.13, -0.01, 0.37, 0.51, 0.06, 0.17,
            0.6, 0.71, 0.67, 0.79, 0.73, 0.85, 0.75, 0.94, 0.93, 0.65, 0.63, 0.71, 1.08, 1.16, 1.17, 0.29, -0.11, 0.41,
            0.41, 0.11, 0.09, 0.61, 0.66, 0.32, 0.74, 0.41, 0.36, 0.86, 0.82, 0.58, 0.94, 0.68, 1.06, 0.76, 0.73, 1.17,
            1.18, -0.07, -0.05, 0.5, 0.03, 0.52, 0.55, 0.22, 0.79, 0.36, 0.83, 0.82, 0.84, 0.86, 0.51, 1., 0.95, 0.75,
            0.73, 1.16, 0.75, 0.78, 0.78, 0.42, 0.04, 0.36, 0.14, 0.9, 0.17, 0.56, 0.8, 0.61, 0.49, 0.66, 0.92, 1.08,
            0.68, 1.13, 0.72, 1.15, 1.15, 0.76, 0.76, 1.18, 0.8]).reshape((1, 11, 11, 2)).astype('float32')

        # expected = tf.image.ssim(y_true, y_pred, max_val=1., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        # expected = self.evaluate(expected)
        # 0.5326015949249268 when compensation = 1

        kernels = _ssim_kernel(size=11, sigma=1.5, channels=2, dtype='float32')
        result, _ = _ssim_level(y_true, y_pred, max_val=1.0, kernels=kernels, k1=0.01, k2=0.03)
        result = tf.reduce_mean(result)
        result = self.evaluate(result)

        self.assertAlmostEqual(0.5322474241256714, result, places=5)


@test_combinations.run_all_keras_modes
class TestStructuralSimilarityLoss(test_combinations.TestCase):
    def test_config(self):
        loss = StructuralSimilarityLoss(reduction=Reduction.NONE, name='loss1')
        self.assertEqual(loss.name, 'loss1')
        self.assertEqual(loss.reduction, Reduction.NONE)

    def test_zeros(self):
        probs = tf.zeros((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'float32')

        result = structural_similarity_loss(
            y_true=targets, y_pred=probs, sample_weight=None, max_val=1., factors=(0.5,), size=2, sigma=1.5,
            k1=0.01, k2=0.03, weight_pooling='mean')
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=1e-4)

    def test_ones(self):
        probs = tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'float32')

        result = structural_similarity_loss(
            y_true=targets, y_pred=probs, sample_weight=None, max_val=1., factors=(0.5,), size=2, sigma=1.5,
            k1=0.01, k2=0.03, weight_pooling='mean')
        result = self.evaluate(result)

        self.assertAllClose(result, [0.] * 3, atol=1e-4)

    def test_false(self):
        probs = tf.zeros((3, 16, 16, 1), 'float32')
        targets = tf.ones((3, 16, 16, 1), 'float32')

        result = structural_similarity_loss(
            y_true=targets, y_pred=probs, sample_weight=None, max_val=1., factors=(0.5,), size=2, sigma=1.5,
            k1=0.01, k2=0.03, weight_pooling='mean')
        result = self.evaluate(result)

        self.assertAllClose(result, [1.] * 3, atol=1e-2)

    def test_true(self):
        probs = tf.ones((3, 16, 16, 1), 'float32')
        targets = tf.zeros((3, 16, 16, 1), 'float32')

        result = structural_similarity_loss(
            y_true=targets, y_pred=probs, sample_weight=None, max_val=1., factors=(0.5,), size=2, sigma=1.5,
            k1=0.01, k2=0.03, weight_pooling='mean')
        result = self.evaluate(result)

        self.assertAllClose(result, [1.] * 3, atol=1e-2)

    def test_even(self):
        probs = tf.zeros((1, 7, 7, 1), 'float32')
        targets = tf.zeros((1, 7, 7, 1), 'float32')

        result = structural_similarity_loss(
            y_true=targets, y_pred=probs, sample_weight=None, max_val=1., factors=(0.5,), size=2, sigma=1.5,
            k1=0.01, k2=0.03, weight_pooling='mean')
        result = self.evaluate(result)

        self.assertAllClose(result, [0.])

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
        ], 'float32', shape=(1, 32, 32, 1)) / 10.
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
        ], 'float32', shape=(1, 32, 32, 1)) / 10.

        # expected = 1. - tf.image.ssim_multiscale(
        #     targets, probs, 1., power_factors=(0.1001, 0.2363, 0.1333), filter_size=5, filter_sigma=1.5,
        #     k1=0.01, k2=0.03)
        # expected = self.evaluate(expected)

        loss = StructuralSimilarityLoss(factors=(0.1001, 0.2363, 0.1333), size=5, sigma=1.5)
        result = loss(targets, probs)
        result = self.evaluate(result)

        self.assertAlmostEqual(result, 0.8972229, places=4)  # 0.8249481320381165 when compensation = 1

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
            [[[0], [1], [1], [0]], [[1], [0], [0], [1]], [[0], [1], [1], [0]], [[1], [1], [1], [1]]]], 'float32')
        logits = tf.repeat(tf.repeat(logits, 16, axis=1), 16, axis=2)
        targets = tf.repeat(tf.repeat(targets, 16, axis=1), 16, axis=2)
        weights = tf.concat([tf.ones((2, 64, 32, 1)), tf.zeros((2, 64, 32, 1))], axis=2)

        loss = StructuralSimilarityLoss(factors=(0.5,), size=2)

        result = self.evaluate(loss(targets[:, :, :32], logits[:, :, :32]))
        self.assertAlmostEqual(result, 0.6233712, places=6)

        result = self.evaluate(loss(targets, logits, weights))
        self.assertAlmostEqual(result, 0.63185704, places=6)

        result = self.evaluate(loss(targets, logits, weights * 2.))
        self.assertAlmostEqual(result, 0.63185704 * 2, places=6)

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

        loss = StructuralSimilarityLoss(factors=(0.5,), size=2)
        result = self.evaluate(loss(targets, logits, weights))

        self.assertAlmostEqual(result, 0.85025084)

    def test_batch(self):
        probs = np.random.rand(2, 224, 224, 1).astype('float32')
        targets = (np.random.rand(2, 224, 224, 1) > 0.5).astype('float32')

        loss = StructuralSimilarityLoss()
        result0 = self.evaluate(loss(targets, probs))
        result1 = sum([self.evaluate(loss(targets[i:i + 1], probs[i:i + 1])) for i in range(2)]) / 2

        self.assertAlmostEqual(result0, result1)

    def test_model(self):
        model = models.Sequential([layers.Dense(1, activation='sigmoid')])
        model.compile(loss='SegMe>Loss>StructuralSimilarityLoss', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.zeros((2, 224, 224, 1)), np.zeros((2, 224, 224, 1), 'int32'))
        models.Sequential.from_config(model.get_config())


if __name__ == '__main__':
    tf.test.main()

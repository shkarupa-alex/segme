import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from segme.utils.common.augs.rotate import rotate, rotate_cw, _rotate, _rotate_cw, _rotate_ccw
from segme.utils.common.augs.tests.testing_utils import aug_samples, max_diff


@test_util.run_all_in_graph_and_eager_modes
class TestRotate(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples('rotate')
        augmented = _rotate(inputs, 45, 'nearest', [[[[0, 128, 255]]]])
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples('rotate', 'float32')
        augmented = _rotate(inputs, 45, 'nearest', [[[[0., 128 / 255, 1.]]]])
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)

    def test_masks_weight(self):
        images = np.random.uniform(high=255, size=[16, 224, 224, 3]).astype('uint8')
        masks = [
            np.random.uniform(high=2, size=[16, 224, 224, 2]).astype('float32'),
            np.random.uniform(high=2, size=[16, 224, 224, 2]).astype('int32')]
        weights = np.random.uniform(size=[16, 224, 224, 3]).astype('float32')

        images_expected = _rotate(images, 45, 'bilinear', [[[[0, 128, 255]]]])
        masks_expected = [_rotate(m, 45, 'nearest', [[[[0, 0]]]]) for m in masks]
        weights_expected = _rotate(weights, 45, 'nearest', [[[[0, 0, 0]]]])

        actual = rotate(images, masks, weights, 0.5, 45, [[[[0, 128, 255]]]])
        images_actual, masks_actual, weights_actual = self.evaluate(actual)
        self.assertSetEqual({0, 1}, set(masks_actual[1].ravel()))

        rotated = tf.reduce_all(tf.equal(images_actual, images_expected), axis=[1, 2, 3])
        rotated = self.evaluate(rotated)
        self.assertIn(True, rotated)
        self.assertIn(False, rotated)

        for i, r in enumerate(rotated):
            if r:
                difference = max_diff(images_actual[i:i + 1], images_expected[i:i + 1])
                difference = self.evaluate(difference)
                self.assertLessEqual(difference, 1e-5)

                for j in range(2):
                    difference = max_diff(masks_actual[j][i:i + 1], masks_expected[j][i:i + 1])
                    difference = self.evaluate(difference)
                    self.assertLessEqual(difference, 1e-5)

                difference = max_diff(weights_actual[i:i + 1], weights_expected[i:i + 1])
                difference = self.evaluate(difference)
                self.assertLessEqual(difference, 1e-5)
            else:
                difference = max_diff(images_actual[i:i + 1], images[i:i + 1])
                difference = self.evaluate(difference)
                self.assertLessEqual(difference, 1e-5)

                for j in range(2):
                    difference = max_diff(masks_actual[j][i:i + 1], masks[j][i:i + 1])
                    difference = self.evaluate(difference)
                    self.assertLessEqual(difference, 1e-5)

                difference = max_diff(weights_actual[i:i + 1], weights[i:i + 1])
                difference = self.evaluate(difference)
                self.assertLessEqual(difference, 1e-5)


@test_util.run_all_in_graph_and_eager_modes
class TestRotateCW(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples('rotate_cw')
        augmented = _rotate_cw(inputs)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples('rotate_cw', 'float32')
        augmented = _rotate_cw(inputs)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)

    def test_some(self):
        inputs = np.random.uniform(high=255, size=[16, 224, 224, 3]).astype('uint8')
        expected = _rotate_cw(inputs)
        augmented, _, _ = rotate_cw(inputs, None, None, .5)
        same = tf.reduce_all(tf.equal(augmented, expected), axis=[1, 2, 3])
        same = self.evaluate(same)
        self.assertIn(True, same)
        self.assertIn(False, same)

    def test_all(self):
        inputs, expected = aug_samples('rotate_cw')
        inputs = tf.image.resize(inputs, [448, 224])
        expected = tf.image.resize(expected, [224, 448])
        augmented, _, _ = rotate_cw(inputs, None, None, 1.)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)


@test_util.run_all_in_graph_and_eager_modes
class TestRotateCCW(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples('rotate_ccw')
        augmented = _rotate_ccw(inputs)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples('rotate_ccw', 'float32')
        augmented = _rotate_ccw(inputs)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)


if __name__ == '__main__':
    tf.test.main()

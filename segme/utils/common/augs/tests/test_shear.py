import tensorflow as tf
from tensorflow.python.framework import test_util
from segme.utils.common.augs.shear import _shear_x, _shear_y
from segme.utils.common.augs.tests.testing_utils import aug_samples, max_diff


@test_util.run_all_in_graph_and_eager_modes
class TestShearX(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples('shear_x')
        augmented = _shear_x(inputs, -0.7, 'bilinear', [[[[0, 128, 255]]]])
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples('shear_x', 'float32')
        augmented = _shear_x(inputs, -0.7, 'bilinear', [[[[0., 128 / 255, 1.]]]])
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1 / 255)


@test_util.run_all_in_graph_and_eager_modes
class TestShearY(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples('shear_y')
        augmented = _shear_y(inputs, 0.7, 'bilinear', [[[[0, 128, 255]]]])
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples('shear_y', 'float32')
        augmented = _shear_y(inputs, 0.7, 'bilinear', [[[[0., 128 / 255, 1.]]]])
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1 / 255)


if __name__ == '__main__':
    tf.test.main()

import tensorflow as tf

from segme.utils.common.augs.shear import _shear_x
from segme.utils.common.augs.shear import _shear_y
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestShearX(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("shear_x")
        augmented = _shear_x(inputs, -0.7, "bilinear", [[[[0, 128, 255]]]])
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("shear_x", "float32")
        augmented = _shear_x(
            inputs, -0.7, "bilinear", [[[[0.0, 128 / 255, 1.0]]]]
        )
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1 / 255)


class TestShearY(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("shear_y")
        augmented = _shear_y(inputs, 0.7, "bilinear", [[[[0, 128, 255]]]])
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("shear_y", "float32")
        augmented = _shear_y(
            inputs, 0.7, "bilinear", [[[[0.0, 128 / 255, 1.0]]]]
        )
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1 / 255)

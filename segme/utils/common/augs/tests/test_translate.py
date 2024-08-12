import tensorflow as tf

from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff
from segme.utils.common.augs.translate import _translate_x
from segme.utils.common.augs.translate import _translate_y


class TestTranslateX(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("translate_x")
        augmented = _translate_x(inputs, -0.7, "bilinear", [[[[0, 128, 255]]]])
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("translate_x", "float32")
        augmented = _translate_x(
            inputs, -0.7, "bilinear", [[[[0.0, 128 / 255, 1.0]]]]
        )
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1 / 255)


class TestTranslateY(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("translate_y")
        augmented = _translate_y(inputs, 0.7, "bilinear", [[[[0, 128, 255]]]])
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("translate_y", "float32")
        augmented = _translate_y(
            inputs, 0.7, "bilinear", [[[[0.0, 128 / 255, 1.0]]]]
        )
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1 / 255)

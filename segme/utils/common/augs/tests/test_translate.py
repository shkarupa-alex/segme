import tensorflow as tf
from tensorflow.python.framework import test_util
from segme.utils.common.augs.translate import _translate_x, _translate_y
from segme.utils.common.augs.tests.testing_utils import aug_samples, max_diff


@test_util.run_all_in_graph_and_eager_modes
class TestTranslateX(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples('translate_x')
        augmented = _translate_x(inputs, -0.7, 'bilinear', [[[[0, 128, 255]]]])
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples('translate_x', 'float32')
        augmented = _translate_x(inputs, -0.7, 'bilinear', [[[[0., 128 / 255, 1.]]]])
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1 / 255)


@test_util.run_all_in_graph_and_eager_modes
class TestTranslateY(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples('translate_y')
        augmented = _translate_y(inputs, 0.7, 'bilinear', [[[[0, 128, 255]]]])
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples('translate_y', 'float32')
        augmented = _translate_y(inputs, 0.7, 'bilinear', [[[[0., 128 / 255, 1.]]]])
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1 / 255)


if __name__ == '__main__':
    tf.test.main()

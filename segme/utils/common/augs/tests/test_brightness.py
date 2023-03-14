import tensorflow as tf
from tensorflow.python.framework import test_util
from segme.utils.common.augs.brightness import _brightness
from segme.utils.common.augs.tests.testing_utils import aug_samples, max_diff


@test_util.run_all_in_graph_and_eager_modes
class TestBrightness(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples('brightness')
        augmented = _brightness(inputs, -0.4)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples('brightness', 'float32')
        augmented = _brightness(inputs, -0.4)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)


if __name__ == '__main__':
    tf.test.main()

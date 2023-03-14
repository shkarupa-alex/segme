import tensorflow as tf
from tensorflow.python.framework import test_util
from segme.utils.common.augs.grayscale import _grayscale
from segme.utils.common.augs.tests.testing_utils import aug_samples, max_diff


@test_util.run_all_in_graph_and_eager_modes
class TestGrayscale(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples('grayscale')
        augmented = _grayscale(inputs, 0.4)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1)

    def test_float(self):
        inputs, expected = aug_samples('grayscale', 'float32')
        augmented = _grayscale(inputs, 0.4)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 2 / 255)


if __name__ == '__main__':
    tf.test.main()

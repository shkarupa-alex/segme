import tensorflow as tf
from tensorflow.python.framework import test_util
from segme.utils.common.augs.flip import _flip_ud, _flip_lr
from segme.utils.common.augs.tests.testing_utils import aug_samples, max_diff


@test_util.run_all_in_graph_and_eager_modes
class TestFlipUD(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples('flip_ud')
        augmented = _flip_ud(inputs)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples('flip_ud', 'float32')
        augmented = _flip_ud(inputs)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)


@test_util.run_all_in_graph_and_eager_modes
class TestFlipLR(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples('flip_lr')
        augmented = _flip_lr(inputs)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples('flip_lr', 'float32')
        augmented = _flip_lr(inputs)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)


if __name__ == '__main__':
    tf.test.main()

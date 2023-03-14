import tensorflow as tf
from tensorflow.python.framework import test_util
from segme.utils.common.augs.mix import _mix
from segme.utils.common.augs.tests.testing_utils import aug_samples, max_diff


@test_util.run_all_in_graph_and_eager_modes
class TestMix(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples('mix')
        augmented = _mix(inputs, 0.4, [[[[0, 128, 255]]]])
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples('mix', 'float32')
        augmented = _mix(inputs, 0.4, [[[[0., 128 / 255, 1.]]]])
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1 / 255)


if __name__ == '__main__':
    tf.test.main()

import tensorflow as tf
from tensorflow.python.framework import test_util
from segme.utils.common.augs.solarize import _solarize
from segme.utils.common.augs.tests.testing_utils import aug_samples, max_diff


@test_util.run_all_in_graph_and_eager_modes
class TestSolarize(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples('solarize')
        augmented = _solarize(inputs)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples('solarize', 'float32')
        augmented = _solarize(inputs)
        difference = max_diff(expected, augmented)
        difference = self.evaluate(difference)
        self.assertLessEqual(difference, 1e-5)


if __name__ == '__main__':
    tf.test.main()

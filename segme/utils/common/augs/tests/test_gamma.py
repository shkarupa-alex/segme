import tensorflow as tf

from segme.utils.common.augs.gamma import _gamma
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestGamma(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("gamma")
        augmented = _gamma(inputs, 0.4, False)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("gamma", "float32")
        augmented = _gamma(inputs, 0.4, False)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1 / 255)

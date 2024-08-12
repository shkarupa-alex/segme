import tensorflow as tf

from segme.utils.common.augs.hue import _hue
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestHue(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("hue")
        augmented = _hue(inputs, -0.4)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("hue", "float32")
        augmented = _hue(inputs, -0.4)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1 / 255)

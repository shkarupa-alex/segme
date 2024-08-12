import tensorflow as tf

from segme.utils.common.augs.contrast import _contrast
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestContrast(tf.test.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("contrast")
        augmented = _contrast(inputs, 1.2)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("contrast", "float32")
        augmented = _contrast(inputs, 1.2)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1 / 255)

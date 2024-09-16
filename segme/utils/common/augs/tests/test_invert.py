from keras.src import testing

from segme.utils.common.augs.invert import _invert
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestInvert(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("invert")
        augmented = _invert(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("invert", "float32")
        augmented = _invert(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

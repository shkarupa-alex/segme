from keras.src import testing

from segme.utils.common.augs.brightness import _brightness
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestBrightness(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("brightness")
        augmented = _brightness(inputs, -0.4)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("brightness", "float32")
        augmented = _brightness(inputs, -0.4)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

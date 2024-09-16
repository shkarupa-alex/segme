from keras.src import testing

from segme.utils.common.augs.saturation import _saturation
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestSaturation(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("saturation")
        augmented = _saturation(inputs, 0.8)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("saturation", "float32")
        augmented = _saturation(inputs, 0.8)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1 / 255)

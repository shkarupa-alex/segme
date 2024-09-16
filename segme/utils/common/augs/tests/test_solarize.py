from keras.src import testing

from segme.utils.common.augs.solarize import _solarize
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestSolarize(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("solarize")
        augmented = _solarize(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("solarize", "float32")
        augmented = _solarize(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

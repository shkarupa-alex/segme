from keras.src import testing

from segme.utils.common.augs.autocontrast import _autocontrast
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestAutoContrast(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("autocontrast")
        augmented = _autocontrast(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1)

    def test_float(self):
        inputs, expected = aug_samples("autocontrast", "float32")
        augmented = _autocontrast(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1 / 255)

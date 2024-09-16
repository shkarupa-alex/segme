from keras.src import testing

from segme.utils.common.augs.blur import _gaussblur
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestGaussblur(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("gaussblur")
        augmented = _gaussblur(inputs, 5)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1)

    def test_float(self):
        inputs, expected = aug_samples("gaussblur", "float32")
        augmented = _gaussblur(inputs, 5)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1 / 255)

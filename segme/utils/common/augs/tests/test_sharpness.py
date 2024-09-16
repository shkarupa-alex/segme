from keras.src import testing

from segme.utils.common.augs.sharpness import _sharpness
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestSharpness(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("sharpness")
        augmented = _sharpness(inputs, 0.4)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1)

    def test_float(self):
        inputs, expected = aug_samples("sharpness", "float32")
        augmented = _sharpness(inputs, 0.4)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 2 / 255)

from keras.src import testing

from segme.utils.common.augs.posterize import _posterize
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestPosterize(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("posterize")
        augmented = _posterize(inputs, 2)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("posterize", "float32")
        augmented = _posterize(inputs, 2)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

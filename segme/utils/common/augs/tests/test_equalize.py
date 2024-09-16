from keras.src import testing

from segme.utils.common.augs.equalize import _equalize
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestEqualize(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("equalize")
        augmented = _equalize(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("equalize", "float32")
        augmented = _equalize(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

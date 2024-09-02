from keras.src import testing

from segme.utils.common.augs.flip import _flip_lr
from segme.utils.common.augs.flip import _flip_ud
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestFlipUD(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("flip_ud")
        augmented = _flip_ud(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("flip_ud", "float32")
        augmented = _flip_ud(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)


class TestFlipLR(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("flip_lr")
        augmented = _flip_lr(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("flip_lr", "float32")
        augmented = _flip_lr(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

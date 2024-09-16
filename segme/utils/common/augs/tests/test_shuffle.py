from keras.src import testing

from segme.utils.common.augs.shuffle import _shuffle
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestShuffle(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("shuffle")
        augmented = _shuffle(inputs, [2, 1, 0])
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("shuffle", "float32")
        augmented = _shuffle(inputs, [2, 1, 0])
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

from keras.src import testing

from segme.utils.common.augs.mix import _mix
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestMix(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("mix")
        augmented = _mix(inputs, 0.4, [[[[0, 128, 255]]]])
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("mix", "float32")
        augmented = _mix(inputs, 0.4, [[[[0.0, 128 / 255, 1.0]]]])
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1 / 255)

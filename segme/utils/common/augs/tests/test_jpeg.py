from keras.src import testing

from segme.utils.common.augs.jpeg import _jpeg
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestJpeg(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("jpeg")
        augmented = _jpeg(inputs, 30)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("jpeg", "float32")
        augmented = _jpeg(inputs, 30)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1 / 255)

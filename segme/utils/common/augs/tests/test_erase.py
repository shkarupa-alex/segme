import numpy as np
import tensorflow as tf

from segme.utils.common.augs.erase import _erase
from segme.utils.common.augs.erase import erase
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestErase(tf.test.TestCase):
    def test_ref(self):
        mask = np.ones([6, 448, 448, 1], "bool")
        for i in range(6):
            mask[i, i * 64 : (i + 1) * 64, i * 64 : (i + 1) * 64] = False

        inputs, expected = aug_samples("erase")
        augmented = _erase(inputs, mask, [[[[0, 128, 255]]]])
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        mask = np.ones([6, 448, 448, 1], "bool")
        for i in range(6):
            mask[i, i * 64 : (i + 1) * 64, i * 64 : (i + 1) * 64] = False

        inputs, expected = aug_samples("erase", "float32")
        augmented = _erase(inputs, mask, [[[[0.0, 128 / 255, 1.0]]]])
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_masks_weight(self):
        images = np.random.uniform(high=255, size=[16, 224, 224, 3]).astype(
            "uint8"
        )
        masks = [
            np.random.uniform(high=2, size=[16, 224, 224, 2]).astype("float32"),
            np.random.uniform(high=2, size=[16, 224, 224, 2]).astype("int32"),
        ]
        weights = np.ones([16, 224, 224, 1], "float32")

        actual = erase(
            images, masks, weights, 0.5, (0.02, 1 / 3), [[[[0, 128, 255]]]]
        )
        images_actual, masks_actual, weights_actual = self.evaluate(actual)
        self.assertSetEqual({0, 1}, set(masks_actual[1].ravel()))

        idx0 = np.where(weights_actual[..., 0] == 0.0)
        self.assertTrue(
            (
                images_actual[idx0].reshape([-1, 3])
                == np.array([[0, 128, 255]], "uint8")
            ).all()
        )

        idx1 = np.where(weights_actual == 1.0)
        self.assertAllEqual(images_actual[idx1], images[idx1])

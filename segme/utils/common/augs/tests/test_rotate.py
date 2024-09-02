import numpy as np
from keras.src import backend
from keras.src import ops
from keras.src import testing

from segme.utils.common.augs.rotate import _rotate
from segme.utils.common.augs.rotate import _rotate_ccw
from segme.utils.common.augs.rotate import _rotate_cw
from segme.utils.common.augs.rotate import rotate
from segme.utils.common.augs.rotate import rotate_cw
from segme.utils.common.augs.tests.testing_utils import aug_samples
from segme.utils.common.augs.tests.testing_utils import max_diff


class TestRotate(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("rotate")
        augmented = _rotate(inputs, 45, "nearest", [[[[0, 128, 255]]]])
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("rotate", "float32")
        augmented = _rotate(inputs, 45, "nearest", [[[[0.0, 128 / 255, 1.0]]]])
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
        weights = np.random.uniform(size=[16, 224, 224, 3]).astype("float32")

        images_expected = _rotate(images, 45, "bilinear", [[[[0, 128, 255]]]])
        masks_expected = [
            _rotate(m, 45, "nearest", [[[[0, 0]]]]) for m in masks
        ]
        weights_expected = _rotate(weights, 45, "nearest", [[[[0, 0, 0]]]])

        images_actual, masks_actual, weights_actual = rotate(
            images, masks, weights, 0.5, 45, [[[[0, 128, 255]]]]
        )
        images_actual = backend.convert_to_numpy(images_actual)
        masks_actual = backend.convert_to_numpy(masks_actual)
        weights_actual = backend.convert_to_numpy(weights_actual)

        self.assertSetEqual({0, 1}, set(masks_actual[1].ravel()))

        rotated = ops.all(
            ops.equal(images_actual, images_expected), axis=[1, 2, 3]
        )
        self.assertIn(True, rotated)
        self.assertIn(False, rotated)

        for i, r in enumerate(rotated):
            if r:
                difference = max_diff(
                    images_actual[i : i + 1], images_expected[i : i + 1]
                )
                self.assertLessEqual(difference, 1e-5)

                for j in range(2):
                    difference = max_diff(
                        masks_actual[j][i : i + 1], masks_expected[j][i : i + 1]
                    )
                    self.assertLessEqual(difference, 1e-5)

                difference = max_diff(
                    weights_actual[i : i + 1], weights_expected[i : i + 1]
                )
                self.assertLessEqual(difference, 1e-5)
            else:
                difference = max_diff(
                    images_actual[i : i + 1], images[i : i + 1]
                )
                self.assertLessEqual(difference, 1e-5)

                for j in range(2):
                    difference = max_diff(
                        masks_actual[j][i : i + 1], masks[j][i : i + 1]
                    )
                    self.assertLessEqual(difference, 1e-5)

                difference = max_diff(
                    weights_actual[i : i + 1], weights[i : i + 1]
                )
                self.assertLessEqual(difference, 1e-5)


class TestRotateCW(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("rotate_cw")
        augmented = _rotate_cw(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("rotate_cw", "float32")
        augmented = _rotate_cw(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_some(self):
        inputs = np.random.uniform(high=255, size=[16, 224, 224, 3]).astype(
            "uint8"
        )
        expected = _rotate_cw(inputs)
        augmented, _, _ = rotate_cw(inputs, None, None, 0.5)
        same = ops.all(ops.equal(augmented, expected), axis=[1, 2, 3])
        self.assertIn(True, same)
        self.assertIn(False, same)

    def test_all(self):
        inputs, expected = aug_samples("rotate_cw")
        inputs = ops.image.resize(inputs, [448, 224])
        expected = ops.image.resize(expected, [224, 448])
        augmented, _, _ = rotate_cw(inputs, None, None, 1.0)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)


class TestRotateCCW(testing.TestCase):
    def test_ref(self):
        inputs, expected = aug_samples("rotate_ccw")
        augmented = _rotate_ccw(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

    def test_float(self):
        inputs, expected = aug_samples("rotate_ccw", "float32")
        augmented = _rotate_ccw(inputs)
        difference = max_diff(expected, augmented)
        self.assertLessEqual(difference, 1e-5)

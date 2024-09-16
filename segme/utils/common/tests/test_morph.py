import cv2
import numpy as np
from keras.src import backend
from keras.src import testing

from segme.utils.common.morph import dilate
from segme.utils.common.morph import erode


class TestErodeDilate(testing.TestCase):
    inputs = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        "uint8",
    )

    def test_erode_3(self):
        inputs = np.pad(self.inputs, 8, mode="constant")
        inputs = np.pad(inputs, 8, mode="edge")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for iterations in range(1, 6):
            expected = cv2.erode(inputs, kernel, iterations=iterations)

            result = erode(
                inputs.astype("int32")[None, ..., None],
                3,
                iterations,
                strict=True,
            )
            result = backend.convert_to_numpy(result)
            result = result.astype("uint8")[0, ..., 0]
            self.assertAlmostEqual(expected, result)

    def test_erode_5(self):
        inputs = np.pad(self.inputs, 8, mode="constant")
        inputs = np.pad(inputs, 8, mode="edge")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        expected = cv2.erode(inputs, kernel, iterations=2)

        result = erode(
            inputs.astype("int32")[None, ..., None], 5, 1, strict=True
        )
        result = backend.convert_to_numpy(result)
        result = result.astype("uint8")[0, ..., 0]
        self.assertAlmostEqual(expected, result)

    def test_dilate_3(self):
        inputs = np.pad(self.inputs, 8, mode="constant")
        inputs = np.pad(inputs, 8, mode="edge")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for iterations in range(1, 6):
            expected = cv2.dilate(inputs, kernel, iterations=iterations)

            result = dilate(
                inputs.astype("int32")[None, ..., None],
                3,
                iterations,
                strict=True,
            )
            result = backend.convert_to_numpy(result)
            result = result.astype("uint8")[0, ..., 0]
            self.assertAlmostEqual(expected, result)

    def test_dilate_5(self):
        inputs = np.pad(self.inputs, 8, mode="constant")
        inputs = np.pad(inputs, 8, mode="edge")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        expected = cv2.dilate(inputs, kernel, iterations=2)

        result = dilate(
            inputs.astype("int32")[None, ..., None], 5, 1, strict=True
        )
        result = backend.convert_to_numpy(result)
        result = result.astype("uint8")[0, ..., 0]
        self.assertAlmostEqual(expected, result)

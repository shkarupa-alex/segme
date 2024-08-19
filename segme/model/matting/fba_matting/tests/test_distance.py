import cv2
import numpy as np
from keras.src import testing

from segme.model.matting.fba_matting.distance import Distance
from segme.model.matting.fba_matting.tests.test_twomap import _twomap


class TestDistance(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            Distance,
            init_kwargs={},
            input_shape=(2, 64, 64, 1),
            input_dtype="uint8",
            expected_output_shape=(2, 64, 64, 6),
            expected_output_dtype="float32",
        )

    def test_value(self):
        trimap = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 0, 128, 0, 0],
                [0, 0, 0, 0, 0, 0, 128, 0, 128, 128, 128, 128, 128, 0, 128, 0],
                [0, 0, 0, 0, 0, 128, 0, 0, 0, 128, 128, 128, 128, 0, 0, 128],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    128,
                    128,
                    128,
                    0,
                    255,
                    128,
                    128,
                    128,
                    128,
                    0,
                    128,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    128,
                    128,
                    128,
                    255,
                    128,
                    128,
                    128,
                    128,
                    0,
                    128,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    128,
                    128,
                    255,
                    255,
                    128,
                    255,
                    255,
                    128,
                    128,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    255,
                    128,
                    255,
                    255,
                    128,
                    255,
                    255,
                    128,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    128,
                    128,
                    128,
                    128,
                    128,
                    128,
                    255,
                    255,
                    128,
                    128,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    128,
                    0,
                    128,
                    128,
                    128,
                    128,
                    128,
                    128,
                    128,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    128,
                    128,
                    0,
                    0,
                    255,
                    128,
                    128,
                    128,
                    255,
                    128,
                    255,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    128,
                    0,
                    255,
                    255,
                    128,
                    255,
                    128,
                    128,
                    128,
                    0,
                ],
                [0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 128, 128, 128, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 128, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 255, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 128, 255, 255, 255, 128, 0, 0, 128, 0],
            ],
            "uint8",
        )
        twomap = _twomap(trimap)
        expected = _distance(twomap)
        expected = np.round(expected * 255.0).astype("uint8")

        result = Distance()(trimap[None, ..., None])[0]

        diff = np.sum(np.abs(result - expected) > 1e-6) / np.prod(result.shape)
        self.assertLess(diff, 0.03)

    def test_single(self):
        for value in [0, 128, 255]:
            trimap = np.full((64, 64), value, dtype="uint8")
            twomap = _twomap(trimap)
            expected = _distance(twomap)
            expected = np.round(expected * 255.0).astype("uint8")

            result = Distance()(trimap[None, ..., None])[0]

            diff = np.sum(np.abs(result - expected) > 1e-6) / np.prod(
                result.shape
            )
            self.assertLess(diff, 0.01)


def _distance(twomap, length=320):
    clicks = np.zeros(twomap.shape[:2] + (6,))
    for k in range(2):
        if np.count_nonzero(twomap[:, :, k]):
            dt_src = 1 - twomap[:, :, k]
            dt_mask = (
                -cv2.distanceTransform(
                    (dt_src * 255).astype(np.uint8), cv2.DIST_L2, 0
                )
                ** 2
            )
            clicks[:, :, 3 * k] = np.exp(dt_mask / (2 * ((0.02 * length) ** 2)))
            clicks[:, :, 3 * k + 1] = np.exp(
                dt_mask / (2 * ((0.08 * length) ** 2))
            )
            clicks[:, :, 3 * k + 2] = np.exp(
                dt_mask / (2 * ((0.16 * length) ** 2))
            )

    return clicks

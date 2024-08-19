import numpy as np
from keras.src import testing

from segme.model.matting.fba_matting.twomap import Twomap


class TestTwomap(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            Twomap,
            init_kwargs={},
            input_shape=(2, 64, 64, 1),
            input_dtype="uint8",
            expected_output_shape=(2, 64, 64, 2),
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
        expected = _twomap(trimap)
        expected = (expected * 255.0).astype("uint8")

        result = Twomap()(trimap[None, ..., None])[0]

        self.assertAllClose(expected, result)


def _twomap(trimap):
    twomap = np.zeros(trimap.shape[:2] + (2,), "float32")
    twomap[trimap / 255.0 == 1, 1] = 1
    twomap[trimap / 255.0 == 0, 0] = 1

    return twomap

import numpy as np
from keras.src import ops
from keras.src import testing

from segme.common.resize import BilinearInterpolation
from segme.common.resize import NearestInterpolation


class TestNearestInterpolation(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            NearestInterpolation,
            init_kwargs={"scale": None},
            input_shape=((2, 16, 16, 10), (2, 24, 32, 3)),
            input_dtype=("float32",) * 2,
            expected_output_shape=(2, 24, 32, 10),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            NearestInterpolation,
            init_kwargs={"scale": 2},
            input_shape=(2, 16, 16, 10),
            input_dtype="float32",
            expected_output_shape=(2, 32, 32, 10),
            expected_output_dtype="float32",
        )

    def test_tile(self):
        inputs = np.random.uniform(size=[3, 1, 1, 5])
        expected = ops.image.resize(inputs, [2, 7], interpolation="nearest")
        result = NearestInterpolation()([inputs, ops.zeros([1, 2, 7, 5])])
        self.assertAllClose(expected, result)


class TestBilinearInterpolation(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            BilinearInterpolation,
            init_kwargs={"scale": None},
            input_shape=((2, 16, 16, 10), (2, 24, 32, 3)),
            input_dtype=("float32",) * 2,
            expected_output_shape=(2, 24, 32, 10),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            BilinearInterpolation,
            init_kwargs={"scale": 2},
            input_shape=(2, 16, 16, 10),
            input_dtype="float32",
            expected_output_shape=(2, 32, 32, 10),
            expected_output_dtype="float32",
        )

    def test_tile(self):
        inputs = np.random.uniform(size=[3, 1, 1, 5])
        expected = ops.image.resize(inputs, [2, 7], interpolation="bilinear")
        result = BilinearInterpolation()([inputs, ops.zeros([1, 2, 7, 5])])
        self.assertAllClose(expected, result)

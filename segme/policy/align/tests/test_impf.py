import tensorflow as tf
from keras.src import testing

from segme.policy.align.impf import ImplicitFeatureAlignment
from segme.policy.align.impf import SpatialEncoding


class TestImplicitFeatureAlignment(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            ImplicitFeatureAlignment,
            init_kwargs={"filters": 12},
            input_shape=((2, 16, 16, 2), (2, 8, 8, 5), (2, 4, 4, 10)),
            input_dtype=("float32",) * 3,
            expected_output_shape=(2, 16, 16, 12),
            expected_output_dtype="float32",
        )


class TestSpatialEncoding(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            SpatialEncoding,
            init_kwargs={"units": 24, "sigma": 6},
            input_shape=(2, 16, 16, 4),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 28),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            SpatialEncoding,
            init_kwargs={"units": 24, "sigma": 4},
            input_shape=(2, 16, 16, 1),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 25),
            expected_output_dtype="float32",
        )

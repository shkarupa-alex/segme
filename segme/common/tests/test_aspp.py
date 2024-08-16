import tensorflow as tf
from keras.src import testing

from segme.common.aspp import AtrousSpatialPyramidPooling


class TestAtrousSpatialPyramidPooling(testing.TestCase):
    

    def test_layer(self):
        self.run_layer_test(
            AtrousSpatialPyramidPooling,
            init_kwargs={"filters": 10, "stride": 8},
            input_shape=(2, 36, 36, 3),
            input_dtype="float32",
            expected_output_shape=(2, 36, 36, 10),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            AtrousSpatialPyramidPooling,
            init_kwargs={"filters": 64, "stride": 16},
            input_shape=(2, 18, 18, 32),
            input_dtype="float32",
            expected_output_shape=(2, 18, 18, 64),
            expected_output_dtype="float32",
        )

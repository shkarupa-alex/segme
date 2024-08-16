import tensorflow as tf
from keras.src import testing

from segme.common.simpool import SimPool


class TestSimPool(testing.TestCase):
    

    def test_layer(self):
        self.run_layer_test(
            SimPool,
            init_kwargs={"num_heads": 1, "qkv_bias": True},
            input_shape=(2, 16, 16, 4),
            input_dtype="float32",
            expected_output_shape=(2, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            SimPool,
            init_kwargs={"num_heads": 4, "qkv_bias": False},
            input_shape=(2, 16, 16, 4),
            input_dtype="float32",
            expected_output_shape=(2, 4),
            expected_output_dtype="float32",
        )

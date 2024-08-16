import tensorflow as tf
from keras.src import testing

from segme.common.mapool import MultiHeadAttentionPooling


class TestMultiheadAttentionPooling(testing.TestCase):
    

    def test_layer(self):
        self.run_layer_test(
            MultiHeadAttentionPooling,
            init_kwargs={"heads": 8, "queries": 1},
            input_shape=(2, 50, 768),
            input_dtype="float32",
            expected_output_shape=(2, 1, 768),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            MultiHeadAttentionPooling,
            init_kwargs={"heads": 8, "queries": 2},
            input_shape=(2, 7, 7, 768),
            input_dtype="float32",
            expected_output_shape=(2, 2, 768),
            expected_output_dtype="float32",
        )

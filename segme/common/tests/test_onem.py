from keras.src import testing

from segme.common.onem import OneMinus


class TestOneMinus(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            OneMinus,
            init_kwargs={},
            input_shape=(2, 8, 8, 4),
            input_dtype="float32",
            expected_output_shape=(2, 8, 8, 4),
            expected_output_dtype="float32",
        )

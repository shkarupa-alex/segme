from keras.src import testing

from segme.common.se import SE


class TestSE(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            SE,
            init_kwargs={"ratio": 0.5},
            input_shape=(2, 4, 4, 3),
            input_dtype="float32",
            expected_output_shape=(2, 4, 4, 3),
            expected_output_dtype="float32",
        )

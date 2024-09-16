from keras.src import testing

from segme.common.inguard import InputGuard


class TestInputGuard(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            InputGuard,
            init_kwargs={},
            input_shape=(2, 4, 4, 3),
            input_dtype="float32",
            expected_output_shape=(2, 4, 4, 3),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            InputGuard,
            init_kwargs={},
            input_shape=(2, 4, 4, 5),
            input_dtype="uint8",
            expected_output_shape=(2, 4, 4, 3),
            expected_output_dtype="uint8",
        )
        self.run_layer_test(
            InputGuard,
            init_kwargs={},
            input_shape=(2, 4, 4, 1),
            input_dtype="uint8",
            expected_output_shape=(2, 4, 4, 3),
            expected_output_dtype="uint8",
        )

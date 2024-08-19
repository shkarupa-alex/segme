from keras.src import testing

from segme.common.gavg import GlobalAverage


class TestGlobalAverage(testing.TestCase):

    def test_layer(self):
        self.run_layer_test(
            GlobalAverage,
            init_kwargs={},
            input_shape=(2, 36, 36, 3),
            input_dtype="float32",
            expected_output_shape=(2, 36, 36, 3),
            expected_output_dtype="float32",
        )

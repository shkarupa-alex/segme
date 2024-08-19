from keras.src import testing

from segme.common.ppm import PyramidPooling


class TestPyramidPooling(testing.TestCase):

    def test_layer(self):
        self.run_layer_test(
            PyramidPooling,
            init_kwargs={"filters": 2, "sizes": (1, 2, 3, 6)},
            input_shape=(2, 18, 18, 3),
            input_dtype="float32",
            expected_output_shape=(2, 18, 18, 2),
            expected_output_dtype="float32",
        )

from keras.src import testing

from segme.common.split import Split


class TestSplit(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            Split,
            init_kwargs={"indices_or_sections": 2, "axis": -1},
            input_shape=(2, 16, 4),
            input_dtype="float32",
            expected_output_shape=((2, 16, 2),) * 2,
            expected_output_dtype=("float32",) * 2,
        )
        self.run_layer_test(
            Split,
            init_kwargs={"indices_or_sections": [2, 8, 16], "axis": 1},
            input_shape=(2, 16, 4),
            input_dtype="float32",
            expected_output_shape=((2, 2, 4), (2, 6, 4), (2, 8, 4), (2, 0, 4)),
            expected_output_dtype=("float32",) * 4,
        )

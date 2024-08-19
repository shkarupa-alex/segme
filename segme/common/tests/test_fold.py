from keras.src import testing

from segme.common.fold import Fold
from segme.common.fold import UnFold


class TestFold(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            Fold,
            init_kwargs={"size": 2},
            input_shape=(2, 8, 8, 4),
            input_dtype="float32",
            expected_output_shape=(2, 4, 4, 16),
            expected_output_dtype="float32",
        )


class TestUnFold(testing.TestCase):

    def test_layer(self):
        self.run_layer_test(
            UnFold,
            init_kwargs={"size": 2},
            input_shape=(2, 8, 8, 4),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 1),
            expected_output_dtype="float32",
        )

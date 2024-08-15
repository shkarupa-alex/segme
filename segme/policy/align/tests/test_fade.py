import tensorflow as tf
from keras.src import testing

from segme.policy.align.fade import FadeFeatureAlignment
from segme.policy.align.fade import SemiShift


class TestFadeFeatureAlignment(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            FadeFeatureAlignment,
            init_kwargs={"filters": 6, "kernel_size": 5, "embedding_size": 16},
            input_shape=((2, 16, 16, 4), (2, 8, 8, 8)),
            input_dtype=("float32",) * 2,
            expected_output_shape=(2, 16, 16, 6),
            expected_output_dtype="float32",
        )


class TestSemiShift(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            SemiShift,
            init_kwargs={"filters": 25, "kernel_size": 3, "embedding_size": 8},
            input_shape=((2, 6, 8, 12), (2, 3, 4, 6)),
            input_dtype=("float32",) * 2,
            expected_output_shape=(2, 6, 8, 25),
            expected_output_dtype="float32",
        )

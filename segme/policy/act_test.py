import unittest

from keras.src import testing

from segme.policy.act import ACTIVATIONS
from segme.policy.act import TLU


class TestActivationsRegistry(unittest.TestCase):
    def test_filled(self):
        self.assertIn("relu", ACTIVATIONS)
        self.assertIn("leakyrelu", ACTIVATIONS)


class TestTLU(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            TLU,
            init_kwargs={},
            input_shape=(2, 8, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 8, 16, 3),
            expected_output_dtype="float32",
        )

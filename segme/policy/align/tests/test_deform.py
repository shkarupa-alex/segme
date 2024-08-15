import tensorflow as tf
from keras.src import testing

from segme.policy.align.deform import DeformableFeatureAlignment
from segme.policy.align.deform import FeatureSelection


class TestDeformableFeatureAlignment(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            DeformableFeatureAlignment,
            init_kwargs={"filters": 8, "deformable_groups": 8},
            input_shape=((2, 16, 16, 6), (2, 8, 8, 12)),
            input_dtype=("float32",) * 2,
            expected_output_shape=(2, 16, 16, 8),
            expected_output_dtype="float32",
        )


class TestFeatureSelection(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            FeatureSelection,
            init_kwargs={"filters": 4},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )

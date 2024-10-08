from keras.src import testing

from segme.policy.align.deform import DCNv2
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


class DCNv2Test(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            DCNv2,
            init_kwargs={
                "filters": 1,
                "kernel_size": 1,
                "strides": 1,
                "padding": "valid",
                "dilation_rate": 1,
                "deformable_groups": 1,
                "use_bias": True,
            },
            input_shape=(2, 3, 4, 2),
            input_dtype="float32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 3, 4, 1),
        )
        self.run_layer_test(
            DCNv2,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "strides": 1,
                "padding": "same",
                "dilation_rate": 1,
                "deformable_groups": 2,
                "use_bias": True,
            },
            input_shape=(2, 3, 4, 3),
            input_dtype="float32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 3, 4, 4),
        )
        self.run_layer_test(
            DCNv2,
            init_kwargs={
                "filters": 2,
                "kernel_size": 3,
                "strides": 1,
                "padding": "same",
                "dilation_rate": 2,
                "deformable_groups": 2,
                "use_bias": True,
            },
            input_shape=(2, 3, 4, 3),
            input_dtype="float32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 3, 4, 2),
        )
        self.run_layer_test(
            DCNv2,
            init_kwargs={
                "filters": 2,
                "kernel_size": 3,
                "strides": 2,
                "padding": "same",
                "dilation_rate": 1,
                "deformable_groups": 2,
                "use_bias": True,
            },
            input_shape=(2, 3, 4, 3),
            input_dtype="float32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 2, 2, 2),
        )
        self.run_layer_test(
            DCNv2,
            init_kwargs={
                "filters": 1,
                "kernel_size": 1,
                "strides": 1,
                "padding": "same",
                "dilation_rate": 1,
                "deformable_groups": 1,
                "use_bias": True,
            },
            input_shape=(2, 1, 1, 3),
            input_dtype="float32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 1, 1, 1),
        )

    def test_custom_alignment(self):
        self.run_layer_test(
            DCNv2,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "strides": 1,
                "padding": "same",
                "dilation_rate": 1,
                "deformable_groups": 2,
                "use_bias": True,
                "custom_alignment": True,
            },
            input_shape=((2, 3, 4, 3), (2, 3, 4, 3)),
            input_dtype=("float32", "float32"),
            expected_output_dtype="float32",
            expected_output_shape=(2, 3, 4, 4),
        )

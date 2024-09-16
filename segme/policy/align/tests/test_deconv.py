from keras.src import testing

from segme.policy.align.deconv import DeconvolutionFeatureAlignment


class TestDeconvolutionFeatureAlignment(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            DeconvolutionFeatureAlignment,
            init_kwargs={"filters": 6, "kernel_size": 4},
            input_shape=((2, 16, 16, 4), (2, 8, 8, 8)),
            input_dtype=("float32",) * 2,
            expected_output_shape=(2, 16, 16, 6),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            DeconvolutionFeatureAlignment,
            init_kwargs={"filters": 6},
            input_shape=((2, 16, 16, 4), (2, 8, 8, 8)),
            input_dtype=("float32",) * 2,
            expected_output_shape=(2, 16, 16, 6),
            expected_output_dtype="float32",
        )

from keras.src import testing

from segme.common.mbconv import MBConv


class TestMBConv(testing.TestCase):

    def test_layer(self):
        self.run_layer_test(
            MBConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "fused": True,
                "strides": 1,
                "expand_ratio": 4.0,
                "se_ratio": 0.0,
                "gamma_initializer": "ones",
                "drop_ratio": 0.0,
            },
            input_shape=(2, 8, 8, 3),
            input_dtype="float32",
            expected_output_shape=(2, 8, 8, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            MBConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "fused": False,
                "strides": 1,
                "expand_ratio": 4.0,
                "se_ratio": 0.0,
                "gamma_initializer": "ones",
                "drop_ratio": 0.0,
            },
            input_shape=(2, 8, 8, 3),
            input_dtype="float32",
            expected_output_shape=(2, 8, 8, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            MBConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "fused": True,
                "strides": 2,
                "expand_ratio": 4.0,
                "se_ratio": 0.0,
                "gamma_initializer": "ones",
                "drop_ratio": 0.0,
            },
            input_shape=(2, 8, 8, 3),
            input_dtype="float32",
            expected_output_shape=(2, 4, 4, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            MBConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "fused": False,
                "strides": 2,
                "expand_ratio": 4.0,
                "se_ratio": 0.0,
                "gamma_initializer": "ones",
                "drop_ratio": 0.0,
            },
            input_shape=(2, 8, 8, 3),
            input_dtype="float32",
            expected_output_shape=(2, 4, 4, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            MBConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "fused": True,
                "strides": 1,
                "expand_ratio": 4.0,
                "se_ratio": 0.2,
                "gamma_initializer": "ones",
                "drop_ratio": 0.0,
            },
            input_shape=(2, 8, 8, 3),
            input_dtype="float32",
            expected_output_shape=(2, 8, 8, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            MBConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "fused": False,
                "strides": 1,
                "expand_ratio": 4.0,
                "se_ratio": 0.2,
                "gamma_initializer": "ones",
                "drop_ratio": 0.0,
            },
            input_shape=(2, 8, 8, 3),
            input_dtype="float32",
            expected_output_shape=(2, 8, 8, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            MBConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "fused": True,
                "strides": 1,
                "expand_ratio": 4.0,
                "se_ratio": 0.0,
                "gamma_initializer": "zeros",
                "drop_ratio": 0.0,
            },
            input_shape=(2, 8, 8, 4),
            input_dtype="float32",
            expected_output_shape=(2, 8, 8, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            MBConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "fused": False,
                "strides": 1,
                "expand_ratio": 4.0,
                "se_ratio": 0.0,
                "gamma_initializer": "zeros",
                "drop_ratio": 0.0,
            },
            input_shape=(2, 8, 8, 4),
            input_dtype="float32",
            expected_output_shape=(2, 8, 8, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            MBConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "fused": True,
                "strides": 1,
                "expand_ratio": 4.0,
                "se_ratio": 0.0,
                "gamma_initializer": "ones",
                "drop_ratio": 0.2,
            },
            input_shape=(2, 8, 8, 4),
            input_dtype="float32",
            expected_output_shape=(2, 8, 8, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            MBConv,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "fused": False,
                "strides": 1,
                "expand_ratio": 4.0,
                "se_ratio": 0.0,
                "gamma_initializer": "ones",
                "drop_ratio": 0.2,
            },
            input_shape=(2, 8, 8, 4),
            input_dtype="float32",
            expected_output_shape=(2, 8, 8, 4),
            expected_output_dtype="float32",
        )

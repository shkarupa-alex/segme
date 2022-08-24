import numpy as np
import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.policy.stdconv import StandardizedConv2D, StandardizedDepthwiseConv2D


@test_combinations.run_all_keras_modes
class TestStandardizedConv2D(test_combinations.TestCase):
    def setUp(self):
        super(TestStandardizedConv2D, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestStandardizedConv2D, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            StandardizedConv2D,
            kwargs={'filters': 4, 'kernel_size': 1, 'strides': 1, 'padding': 'valid'},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            StandardizedConv2D,
            kwargs={'filters': 4, 'kernel_size': 3, 'strides': 2, 'padding': 'same'},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))

    def test_value(self):
        inputs = np.array([
            0.298, -0.336, 4.725, -0.393, -1.951, -3.269, -2.453, -0.043, 0.25, -4.571, -2.143, 2.744, 1.483, 3.229,
            1.472, -1.802, 3.146, -0.048, 1.407, 1.315, -0.823, 0.763, -0.103, 0.295, 1.507, -3.52, -1.55, -2.573,
            0.929, 1.649, 1.545, -0.365, 1.845, 1.208, -0.829, -3.652], 'float32').reshape((1, 3, 4, 3))
        kernel = np.array([
            0.307, -0.094, 0.031, -0.301, -0.164, -0.073, 0.07, -0.167, 0.267, -0.128, -0.226, -0.181, -0.248, -0.05,
            0.056, -0.535, 0.221, -0.04, 0.521, -0.285, -0.323, 0.094, 0.362, -0.022, -0.097, -0.054, -0.084],
            'float32').reshape((3, 3, 3, 1))
        bias = np.zeros((1,), 'float32')
        expected = np.array([7.993624, -9.57225], 'float32').reshape((1, 1, 2, 1))

        layer = StandardizedConv2D(1, 3)
        layer.build(inputs.shape)
        layer.set_weights([kernel, bias])

        result = layer(inputs, training=True)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

        result = layer(inputs)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)


@test_combinations.run_all_keras_modes
class TestStandardizedDepthwiseConv2D(test_combinations.TestCase):
    def setUp(self):
        super(TestStandardizedDepthwiseConv2D, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestStandardizedDepthwiseConv2D, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            StandardizedDepthwiseConv2D,
            kwargs={'kernel_size': 1, 'strides': 1, 'padding': 'valid'},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 8],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        result = test_utils.layer_test(
            StandardizedDepthwiseConv2D,
            kwargs={'kernel_size': 3, 'strides': 2, 'padding': 'same'},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 8],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == '__main__':
    tf.test.main()

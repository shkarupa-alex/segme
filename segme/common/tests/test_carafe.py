import numpy as np
import tensorflow as tf
from tf_keras import mixed_precision
from tf_keras.src.testing_infra import test_combinations
from segme.common.carafe import CarafeConvolution
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestCarafeConvolution(test_combinations.TestCase):
    def setUp(self):
        super(TestCarafeConvolution, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestCarafeConvolution, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            CarafeConvolution,
            kwargs={'kernel_size': 3},
            input_shapes=[(2, 3, 4, 6), (2, 6, 8, 18)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 6, 8, 6)],
            expected_output_dtypes=['float32']
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            CarafeConvolution,
            kwargs={'kernel_size': 3},
            input_shapes=[(2, 3, 4, 6), (2, 6, 8, 18)],
            input_dtypes=['float16'] * 2,
            expected_output_shapes=[(None, 6, 8, 6)],
            expected_output_dtypes=['float16']
        )

    def test_value(self):
        features = np.array([
            0.28, 0.86, 0.86, 0.92, 0.91, 0.67, 0.95, 0.36, 0.11, 0.38, 0.53, 0.36, 0.1, 0.26, 0.09, 0.94, 0.07, 0.75,
            0.05, 0.57, 0.43, 0.13, 0.44, 0.22, 0.65, 0.4, 0.27, 0.45, 0.89, 0.71, 0.14, 0.48, 0.28, 0.51, 0.11, 0.77,
            0.91, 0.11, 0.55, 0.55, 0.67, 0.89, 0.24, 0.16, 0.76, 0.31, 0.98, 0.6
        ]).reshape((2, 3, 4, 2)).astype('float32')
        masks = np.array([
            0.52, 0.84, 0.19, 0.24, 0.04, 0.96, 0.53, 1.0, 0.08, 0.25, 0.7, 0.95, 0.48, 0.56, 0.21, 0.72, 0.23, 0.69,
            0.0, 0.59, 0.98, 0.32, 0.0, 0.41, 0.37, 0.15, 0.39, 0.61, 0.15, 0.15, 0.37, 0.59, 0.2, 0.56, 0.9, 0.94,
            0.33, 0.1, 0.53, 0.39, 0.58, 0.47, 0.15, 0.96, 0.14, 0.41, 0.74, 0.09, 0.35, 0.29, 0.62, 0.44, 0.49, 0.03,
            0.53, 0.47, 0.71, 0.3, 0.68, 0.66, 0.49, 0.59, 0.72, 0.62, 0.57, 0.05, 0.95, 0.7, 0.17, 0.39, 0.0, 0.11,
            0.44, 0.05, 1.0, 0.07, 0.61, 0.98, 0.17, 0.42, 0.3, 0.35, 0.31, 0.96, 0.44, 0.88, 0.49, 0.65, 0.14, 0.84,
            0.3, 0.7, 0.82, 0.73, 0.07, 0.4, 0.19, 0.08, 0.1, 0.44, 0.07, 0.06, 0.11, 0.37, 0.9, 0.03, 0.93, 0.64, 0.85,
            0.64, 0.83, 0.86, 0.14, 0.72, 0.75, 0.92, 0.99, 0.82, 0.61, 0.39, 0.77, 0.09, 0.78, 0.93, 0.17, 0.43, 0.8,
            0.39, 0.47, 0.7, 0.28, 0.13, 0.78, 0.79, 0.78, 1.0, 0.72, 0.93, 0.94, 0.61, 0.62, 0.3, 0.46, 0.35, 0.64,
            0.74, 0.16, 0.23, 0.07, 0.69, 0.21, 0.09, 0.31, 0.93, 0.84, 0.64, 0.72, 0.53, 0.77, 0.18, 0.91, 0.62, 0.78,
            0.4, 0.28, 0.69, 0.12, 0.84, 0.85, 0.66, 0.32, 0.27, 0.05, 0.69, 0.48, 0.01, 0.38, 1.0, 0.73, 0.98, 0.96,
            0.93, 0.7, 0.06, 0.36, 0.14, 0.83, 0.87, 0.77, 0.71, 0.43, 0.01, 0.69, 0.38, 0.74, 0.67, 0.65, 0.3, 0.45,
            0.08, 0.21, 0.73, 0.48, 0.41, 0.85, 0.68, 0.28, 0.74, 0.93, 0.78, 0.97, 0.68, 0.9, 0.17, 0.63, 0.82, 0.91,
            0.15, 0.59, 0.56, 0.41, 0.27, 0.34, 0.95, 0.62, 0.78, 0.93, 0.35, 0.65, 0.63, 0.92, 0.08, 0.28, 0.75, 0.91,
            0.56, 0.41, 0.25, 0.65, 0.86, 0.46, 0.81, 0.52, 0.48, 0.88, 0.49, 0.52, 0.1, 0.4, 0.06, 0.49, 0.22, 0.86,
            0.51, 0.28, 0.28, 0.71, 0.34, 0.21, 0.42, 0.85, 0.98, 0.61, 0.24, 0.72, 0.22, 0.04, 0.12, 0.78, 0.13, 0.61,
            0.63, 0.26, 0.52, 0.4, 0.58, 0.3, 0.21, 0.62, 0.3, 0.25, 0.01, 0.5, 0.3, 0.04, 0.54, 0.34, 0.43, 0.91, 0.72,
            0.79, 0.56, 0.22, 0.93, 0.48, 0.25, 0.18, 0.75, 0.43, 0.71, 0.19, 0.36, 0.88, 0.59, 0.14, 0.18, 0.29, 0.75,
            0.8, 0.16, 0.92, 0.89, 0.99, 0.35, 0.7, 0.79, 0.64, 0.87, 0.36, 0.0, 0.12, 0.65, 0.66, 1.0, 0.7, 0.32, 0.38,
            0.61, 0.92, 0.84, 0.8, 0.76, 0.63, 0.37, 0.28, 0.49, 0.19, 0.62, 0.29, 0.64, 0.78, 0.73, 0.38, 0.58, 0.34,
            0.3, 0.12, 0.36, 0.46, 0.83, 0.23, 0.04, 0.71, 0.85, 0.53, 0.68, 0.69, 0.42, 0.04, 0.64, 0.86, 0.81, 0.31,
            0.74, 0.13, 0.02, 0.53, 0.63, 0.84, 0.23, 0.61, 1.0, 0.68, 0.65, 0.66, 0.18, 0.26, 0.11, 0.84, 0.85, 0.7,
            0.01, 0.04, 0.73, 0.81, 0.04, 0.33, 0.84, 0.33, 0.25, 0.83, 0.19, 0.27, 0.87, 0.08, 0.65, 0.01, 0.17, 0.43,
            0.68, 0.44, 0.16, 0.48, 0.58, 0.84, 0.26, 0.61, 0.23, 0.65, 0.83, 0.84, 0.36, 0.72, 0.65, 0.77, 0.13, 0.97,
            0.78, 0.19, 0.36, 0.79, 0.85, 0.16, 0.1, 0.51, 0.88, 0.84, 0.12, 0.13, 0.47, 0.7, 0.16, 0.99, 0.9, 0.42,
            0.8, 0.36, 0.5, 0.24, 0.09, 0.25, 0.15, 0.93, 0.86, 0.68, 0.73, 0.86, 0.25, 0.38, 1.0, 0.38, 0.11, 0.79,
            0.72, 0.17, 0.69, 0.21, 0.26, 0.82, 0.29, 0.84, 0.98, 0.43, 0.35, 0.66, 0.9, 0.18, 0.56, 0.36, 0.54, 0.54,
            0.64, 0.95, 0.57, 0.25, 0.21, 0.91, 0.96, 0.75, 0.35, 0.88, 0.68, 0.02, 0.25, 0.63, 0.63, 0.29, 0.29, 0.16,
            0.42, 0.73, 0.97, 0.94, 0.82, 0.57, 0.86, 0.26, 0.42, 0.75, 0.69, 0.49, 0.66, 0.98, 0.15, 0.38, 0.04, 0.22,
            0.92, 0.49, 0.71, 0.57, 0.23, 0.71, 0.63, 0.42, 0.81, 0.92, 0.31, 0.57, 0.64, 0.74, 0.42, 0.98, 0.09, 0.96,
            0.07, 0.96, 0.85, 0.53, 0.31, 0.29, 0.69, 0.65, 0.49, 0.89, 0.99, 0.63, 0.59, 0.31, 0.24, 0.08, 0.04, 0.51,
            0.03, 0.66, 0.88, 0.62, 0.12, 0.39, 0.99, 0.2, 0.96, 0.45, 0.54, 0.63, 0.56, 0.36, 0.56, 0.93, 0.12, 0.13,
            0.99, 0.7, 0.93, 0.24, 0.75, 0.11, 0.13, 0.2, 0.82, 0.07, 0.25, 0.77, 0.68, 0.47, 0.7, 0.36, 0.97, 0.84,
            0.11, 0.42, 0.51, 0.08, 0.35, 0.57, 0.51, 0.65, 0.94, 0.13, 0.22, 0.97, 0.94, 0.58, 0.67, 0.19, 0.98, 0.38,
            0.68, 0.97, 0.74, 0.13, 0.08, 0.45, 0.72, 0.92, 0.78, 0.47, 0.56, 0.28, 0.61, 0.42, 0.62, 0.98, 0.45, 0.83,
            0.47, 0.95, 0.3, 0.85, 0.37, 0.46, 0.63, 0.66, 0.03, 0.41, 0.31, 0.15, 0.42, 0.22, 0.52, 0.85, 0.96, 0.64,
            0.79, 0.54, 0.54, 0.65, 0.9, 0.9, 0.11, 0.48, 0.13, 0.99, 0.74, 0.97, 0.57, 0.18, 0.19, 0.36, 0.13, 0.84,
            0.71, 0.73, 0.01, 0.46, 0.64, 0.52, 0.66, 0.51, 0.95, 0.54, 0.48, 0.64, 0.5, 0.11, 0.72, 0.43, 0.87, 0.46,
            0.62, 0.84, 0.04, 0.72, 0.18, 0.53, 0.74, 0.98, 0.01, 0.83, 0.51, 0.03, 0.69, 0.72, 0.2, 0.06, 0.37, 0.8,
            0.03, 0.48, 0.2, 0.09, 0.02, 0.64, 0.28, 0.94, 0.19, 0.67, 0.69, 0.1, 0.61, 0.38, 0.56, 0.22, 0.37, 0.81,
            0.51, 0.68, 0.18, 0.36, 0.99, 0.69, 0.74, 0.32, 0.08, 0.09, 0.1, 0.48, 0.87, 0.81, 0.11, 0.74, 0.73, 0.64,
            0.75, 0.87, 0.57, 0.34, 0.45, 0.28, 0.63, 0.52, 0.3, 0.06, 0.08, 0.55, 0.62, 0.49, 0.06, 0.57, 0.05, 0.32,
            0.31, 0.07, 0.86, 0.37, 0.43, 0.14, 0.21, 0.76, 0.42, 0.41, 0.49, 0.86, 0.13, 0.06, 0.04, 0.99, 0.87, 0.82,
            0.63, 0.33, 0.79, 0.26, 0.37, 0.99, 0.22, 0.81, 0.79, 0.03, 0.8, 0.18, 0.35, 0.14, 0.55, 0.88, 0.83, 0.06,
            0.01, 0.57, 0.68, 0.43, 0.2, 0.4, 0.22, 0.62, 0.3, 0.14, 1.0, 0.95, 0.65, 0.49, 0.23, 0.43, 0.04, 0.35,
            0.39, 0.85, 0.19, 0.99, 0.19, 0.07, 0.43, 0.94, 0.64, 0.49, 0.53, 0.06, 0.21, 0.38, 0.29, 0.36, 0.22, 0.04,
            0.88, 0.65, 0.47, 0.13, 0.09, 0.52, 0.58, 0.02, 0.53, 0.68, 0.13, 0.09, 0.28, 0.73, 0.14, 0.33, 0.98, 0.95,
            0.36, 0.22, 0.58, 0.31, 0.19, 0.73, 0.73, 0.5, 0.69, 0.96, 0.58, 0.28, 0.61, 0.58, 0.61, 0.94, 0.96, 0.06,
            0.09, 0.1, 0.84, 0.94, 0.35, 0.15, 0.1, 0.99, 0.09, 0.82, 0.63, 0.47, 0.46, 0.68, 0.75
        ]).reshape((2, 6, 8, 9)).astype('float32')
        expected = np.array([
            0.2182, 0.302, 0.1739, 0.243, 0.2634, 0.3323, 0.3129, 0.3855, 0.3859, 0.3798, 0.3904, 0.3565, 0.2132,
            0.2323, 0.3204, 0.269, 0.2467, 0.3409, 0.2097, 0.296, 0.2691, 0.359, 0.384, 0.4048, 0.3423, 0.3929, 0.4008,
            0.379, 0.2145, 0.2731, 0.2431, 0.2354, 0.2212, 0.4252, 0.2131, 0.4441, 0.3108, 0.5556, 0.3386, 0.5255,
            0.5308, 0.4961, 0.4569, 0.5206, 0.3182, 0.2879, 0.317, 0.2823, 0.1897, 0.4224, 0.2161, 0.4305, 0.3515,
            0.5492, 0.4052, 0.5775, 0.4985, 0.4936, 0.5476, 0.5054, 0.3388, 0.291, 0.3315, 0.2914, 0.0982, 0.2415,
            0.0986, 0.2528, 0.1532, 0.2497, 0.1277, 0.2366, 0.1986, 0.2445, 0.1667, 0.2423, 0.101, 0.1812, 0.1214,
            0.1826, 0.0683, 0.1725, 0.1229, 0.2765, 0.1752, 0.2499, 0.1254, 0.2827, 0.1466, 0.2616, 0.189, 0.2468,
            0.1085, 0.1324, 0.1204, 0.1651, 0.1167, 0.1831, 0.1503, 0.2638, 0.4039, 0.3137, 0.3321, 0.3007, 0.3531,
            0.374, 0.3275, 0.3136, 0.3132, 0.2162, 0.323, 0.248, 0.1459, 0.252, 0.126, 0.2132, 0.3172, 0.3173, 0.3411,
            0.3314, 0.3214, 0.3935, 0.2812, 0.3185, 0.3021, 0.2527, 0.2586, 0.168, 0.289, 0.3811, 0.2428, 0.3715,
            0.4908, 0.4839, 0.5294, 0.4968, 0.5118, 0.4271, 0.5538, 0.4729, 0.4795, 0.3194, 0.4743, 0.2769, 0.246,
            0.3488, 0.2269, 0.3349, 0.4932, 0.5235, 0.468, 0.4867, 0.558, 0.4635, 0.5731, 0.4687, 0.4806, 0.3087,
            0.5437, 0.3516, 0.1691, 0.2961, 0.132, 0.2353, 0.3679, 0.3031, 0.3786, 0.3503, 0.4045, 0.3088, 0.4502,
            0.2762, 0.4033, 0.1964, 0.3623, 0.2002, 0.1647, 0.2786, 0.1608, 0.3085, 0.3522, 0.3505, 0.3458, 0.3138,
            0.4469, 0.284, 0.3612, 0.2745, 0.3575, 0.1712, 0.3709, 0.2104]).reshape((2, 6, 8, 2)).astype('float32')

        result = CarafeConvolution(3)([features, masks])
        result = self.evaluate(result)

        self.assertAllClose(expected, result, atol=1e-4)


if __name__ == '__main__':
    tf.test.main()

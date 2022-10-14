import numpy as np
import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.common.ffc import FourierUnit, SpectralTransform, FastFourierConv
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestFourierUnit(test_combinations.TestCase):
    def setUp(self):
        super(TestFourierUnit, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestFourierUnit, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            FourierUnit,
            kwargs={'filters': 4},
            input_shape=[2, 4, 6, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 6, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            FourierUnit,
            kwargs={'filters': 2},
            input_shape=[2, 4, 5, 3],
            input_dtype='float16',
            expected_output_shape=[None, 4, 5, 2],
            expected_output_dtype='float16'
        )

    def test_value(self):
        inputs = np.array([
            0.531, -1.039, -0.999, 0.097, 0.096, -4.156, -4.551, 2.614, -1.148, 5.114, -0.416, 1.99, 0.229, 1.913,
            3.671, -4.692, -3.357, -2.9, 1.867, 2.835, 0.65, -0.257, 0.637, 3.234, -0.03, 2.669, -3.037, 0.555, 3.946,
            -1.935, 4.573, -0.169, 1.956, -0.285, 3.56, 0.315, -0.021, 0.619, -5.293, 3.374, 1.504, -0.973, 1.528,
            2.164, -0.535, 1.612, 1.161, -0.075, -1.456, -2.346, 3.226, 1.625, 1.258, 3.023, 1.891, -0.225, -0.134,
            -0.584, -0.644, -3.612, 0.28, -2.674, -3.944, 1.25, -1.555, 2.857, -3.639, -0.383, -0.34, 0.928, -1.014,
            2.511, -2.537, 1.036, -2.608, 0.769, -2.018, 1.833, -1.053, 1.917, -1.686, 3.579, 2.125, 0.044, 1.064,
            1.578, 0.749, 3.132, 2.213, -0.883, -0.505, -0.17, 3.733, -0.024, -0.013, 2.096, -5.058, -0.072, 0.339,
            2.594, 0.731, -3.35, 0.34, 1.289, 3.109, 3.199, -0.201, 1.848, -0.217, -0.142, -3.85, -1.61, -0.66, 1.127,
            -0.019, 2.017, 0.626, -0.393, 0.572, 1.457], 'float32').reshape((2, 4, 5, 3))
        kernel = np.array([
            0.14823073, -0.2577892, -0.29297718, 0.20172161, 0.36700642, -0.22442622, -0.18774046, -0.26406044,
            0.03650609, 0.04282019, 0.3550893, 0.07211816, -0.10894078, -0.05426875, 0.00090706, 0.25713605, 0.2016992,
            0.32344055, 0.20251119, 0.40381187, -0.20418897, 0.13839805, -0.29796243, 0.36498207],
            'float32').reshape((1, 1, 6, 4))
        expected = np.array([  # channels order around conv-norm-act is different from original implementation
            1.03072, 2.07298112, -0.48757631, -3.25151682, 1.11899638, 0.17957781, 0.31252247, 0.28591585, 0.89986151,
            2.30037642, -0.1778176, -0.90312082, 0.88388819, 0.54588276, 0.06446964, 1.13660586, 0.60332996,
            -1.51997876, 1.43986368, 0.57510895, 0.64166951, -1.00020063, 0.20346469, -0.23976682, 0.10744386,
            -1.78121138, 0.97573268, 1.07922494, 0.55584216, 0.35461971, 0.01685912, 1.97590137, -0.69701618,
            0.55142349, -0.05349084, 0.09766511, -0.10614236, 0.04921505, 0.2245338, -2.50870299, 1.29783976,
            1.08606386, -1.15438068, -0.74981588, 0.34352446, 0.1115455, -0.47119746, 0.75082779, 0.53260404,
            1.05368853, 0.0054566, -0.00683769, -0.30345982, -0.84580112, 1.25131834, -0.62564349, -0.2402896,
            0.14729242, -0.14848346, 0.74193341, 0.21708426, 0.74533498, 0.12543491, -0.79470146, -0.78848773,
            -0.25459895, 0.4523361, 0.53146398, 0.54202247, 0.64378113, -1.0816685, -0.14992011, 0.88340777,
            -1.35051453, -0.04230087, -0.99512309, 0.37252092, -0.34271353, 0.40027899, 0.30373785
        ], 'float32').reshape((2, 4, 5, 2))

        layer = FourierUnit(2)
        layer.build(inputs.shape)
        layer.cna.build([2, 4, 3, 6])
        layer.set_weights([kernel] + layer.weights[1:])

        result = layer(inputs)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)


@test_combinations.run_all_keras_modes
class TestSpectralTransform(test_combinations.TestCase):
    def setUp(self):
        super(TestSpectralTransform, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSpectralTransform, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SpectralTransform,
            kwargs={'filters': 8, 'strides': 1, 'use_bias': False, 'use_lfu': True},
            input_shape=[2, 4, 6, 5],
            input_dtype='float32',
            expected_output_shape=[None, 4, 6, 8],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SpectralTransform,
            kwargs={'filters': 8, 'strides': 2, 'use_bias': False, 'use_lfu': True},
            input_shape=[2, 8, 12, 5],
            input_dtype='float32',
            expected_output_shape=[None, 4, 6, 8],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SpectralTransform,
            kwargs={'filters': 8, 'strides': 1, 'use_bias': True, 'use_lfu': True},
            input_shape=[2, 4, 6, 5],
            input_dtype='float32',
            expected_output_shape=[None, 4, 6, 8],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SpectralTransform,
            kwargs={'filters': 8, 'strides': 1, 'use_bias': False, 'use_lfu': False},
            input_shape=[2, 4, 6, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 6, 8],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            SpectralTransform,
            kwargs={'filters': 8, 'strides': 2, 'use_lfu': True},
            input_shape=[2, 8, 12, 5],
            input_dtype='float16',
            expected_output_shape=[None, 4, 6, 8],
            expected_output_dtype='float16'
        )


@test_combinations.run_all_keras_modes
class TestFastFourierConv(test_combinations.TestCase):
    def setUp(self):
        super(TestFastFourierConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestFastFourierConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            FastFourierConv,
            kwargs={
                'filters': 32, 'kernel_size': 3, 'ratio': 0.25, 'strides': 1, 'dilation_rate': 1, 'use_bias': False,
                'use_lfu': True},
            input_shapes=[(2, 16, 16, 16), (2, 16, 16, 8)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 16, 16, 24), (None, 16, 16, 8)],
            expected_output_dtypes=['float32'] * 2
        )
        layer_multi_io_test(
            FastFourierConv,
            kwargs={
                'filters': 32, 'kernel_size': 3, 'ratio': 0., 'strides': 1, 'dilation_rate': 1, 'use_bias': False,
                'use_lfu': True},
            input_shapes=[(2, 16, 16, 16), (2, 16, 16, 8)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 16, 16, 32)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            FastFourierConv,
            kwargs={
                'filters': 32, 'kernel_size': 3, 'ratio': 0.25, 'strides': 2, 'dilation_rate': 1, 'use_bias': False,
                'use_lfu': True},
            input_shapes=[(2, 16, 16, 16), (2, 16, 16, 8)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 8, 8, 24), (None, 8, 8, 8)],
            expected_output_dtypes=['float32'] * 2
        )
        layer_multi_io_test(
            FastFourierConv,
            kwargs={
                'filters': 32, 'kernel_size': 3, 'ratio': 0.25, 'strides': 1, 'dilation_rate': 2, 'use_bias': False,
                'use_lfu': True},
            input_shapes=[(2, 16, 16, 16), (2, 16, 16, 8)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 16, 16, 24), (None, 16, 16, 8)],
            expected_output_dtypes=['float32'] * 2
        )
        layer_multi_io_test(
            FastFourierConv,
            kwargs={
                'filters': 32, 'kernel_size': 3, 'ratio': 0.25, 'strides': 1, 'dilation_rate': 1, 'use_bias': True,
                'use_lfu': True},
            input_shapes=[(2, 16, 16, 16), (2, 16, 16, 8)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 16, 16, 24), (None, 16, 16, 8)],
            expected_output_dtypes=['float32'] * 2
        )
        layer_multi_io_test(
            FastFourierConv,
            kwargs={
                'filters': 32, 'kernel_size': 3, 'ratio': 0.25, 'strides': 1, 'dilation_rate': 1, 'use_bias': False,
                'use_lfu': False},
            input_shapes=[(2, 16, 16, 16), (2, 16, 16, 8)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 16, 16, 24), (None, 16, 16, 8)],
            expected_output_dtypes=['float32'] * 2
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            FastFourierConv,
            kwargs={
                'filters': 32, 'kernel_size': 3, 'ratio': 0.25, 'strides': 1, 'dilation_rate': 1, 'use_bias': False,
                'use_lfu': True},
            input_shapes=[(2, 16, 16, 16), (2, 16, 16, 8)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 16, 16, 24), (None, 16, 16, 8)],
            expected_output_dtypes=['float16'] * 2
        )


if __name__ == '__main__':
    tf.test.main()

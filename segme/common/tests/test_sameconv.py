import numpy as np
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from keras.mixed_precision import policy as mixed_precision
from ..sameconv import SameConv, SameStandardizedConv, SameDepthwiseConv, SameStandardizedDepthwiseConv


@keras_parameterized.run_all_keras_modes
class TestSameConv(keras_parameterized.TestCase):
    def setUp(self):
        super(TestSameConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSameConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            SameConv,
            kwargs={'filters': 4, 'kernel_size': 1, 'strides': 1},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

        mixed_precision.set_global_policy('mixed_float16')
        result = testing_utils.layer_test(
            SameConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'strides': 2},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))


@keras_parameterized.run_all_keras_modes
class TestSameStandardizedConv(keras_parameterized.TestCase):
    def setUp(self):
        super(TestSameStandardizedConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSameStandardizedConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            SameStandardizedConv,
            kwargs={'filters': 4, 'kernel_size': 1, 'strides': 1},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

        mixed_precision.set_global_policy('mixed_float16')
        result = testing_utils.layer_test(
            SameStandardizedConv,
            kwargs={'filters': 4, 'kernel_size': 3, 'strides': 2},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 4],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))


@keras_parameterized.run_all_keras_modes
class TestSameDepthwiseConv(keras_parameterized.TestCase):
    def setUp(self):
        super(TestSameDepthwiseConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSameDepthwiseConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            SameDepthwiseConv,
            kwargs={'kernel_size': 1, 'strides': 1},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 8],
            expected_output_dtype='float32'
        )

        mixed_precision.set_global_policy('mixed_float16')
        result = testing_utils.layer_test(
            SameDepthwiseConv,
            kwargs={'kernel_size': 3, 'strides': 2},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 8],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))


@keras_parameterized.run_all_keras_modes
class TestSameStandardizedDepthwiseConv(keras_parameterized.TestCase):
    def setUp(self):
        super(TestSameStandardizedDepthwiseConv, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSameStandardizedDepthwiseConv, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            SameStandardizedDepthwiseConv,
            kwargs={'kernel_size': 1, 'strides': 1},
            input_shape=[2, 16, 16, 8],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 8],
            expected_output_dtype='float32'
        )

        mixed_precision.set_global_policy('mixed_float16')
        result = testing_utils.layer_test(
            SameStandardizedDepthwiseConv,
            kwargs={'kernel_size': 3, 'strides': 2},
            input_shape=[2, 16, 16, 8],
            input_dtype='float16',
            expected_output_shape=[None, 8, 8, 8],
            expected_output_dtype='float16'
        )
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == '__main__':
    tf.test.main()
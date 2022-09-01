import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.common.pad import SymmetricPadding, SamePadding


@test_combinations.run_all_keras_modes
class TestSymmetricPadding(test_combinations.TestCase):
    def setUp(self):
        super(TestSymmetricPadding, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSymmetricPadding, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SymmetricPadding,
            kwargs={'padding': 1},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 6, 7, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            SymmetricPadding,
            kwargs={'padding': 1},
            input_shape=[2, 4, 5, 3],
            input_dtype='float16',
            expected_output_shape=[None, 6, 7, 3],
            expected_output_dtype='float16'
        )

    def test_error(self):
        with self.assertRaisesRegex(ValueError, 'Symmetric padding can lead to misbehavior'):
            SymmetricPadding(((0, 1), (1, 2)))


@test_combinations.run_all_keras_modes
class TestSamePadding(test_combinations.TestCase):
    def setUp(self):
        super(TestSamePadding, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSamePadding, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SamePadding,
            kwargs={'kernel_size': 1, 'strides': 1, 'dilation_rate': 1, 'symmetric_pad': False},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 5, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SamePadding,
            kwargs={'kernel_size': 1, 'strides': 2, 'dilation_rate': 1, 'symmetric_pad': False},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 5, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SamePadding,
            kwargs={'kernel_size': 1, 'strides': 1, 'dilation_rate': 2, 'symmetric_pad': False},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 5, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SamePadding,
            kwargs={'kernel_size': 1, 'strides': 1, 'dilation_rate': 1, 'symmetric_pad': True},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 5, 3],
            expected_output_dtype='float32'
        )

        test_utils.layer_test(
            SamePadding,
            kwargs={'kernel_size': 3, 'strides': 1, 'dilation_rate': 1, 'symmetric_pad': False},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 5, 3],
            expected_output_dtype='float32'
        )

        test_utils.layer_test(
            SamePadding,
            kwargs={'kernel_size': 3, 'strides': 2, 'dilation_rate': 1, 'symmetric_pad': False},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 6, 7, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SamePadding,
            kwargs={'kernel_size': 3, 'strides': 1, 'dilation_rate': 2, 'symmetric_pad': False},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 9, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SamePadding,
            kwargs={'kernel_size': 3, 'strides': 1, 'dilation_rate': 1, 'symmetric_pad': True},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 6, 7, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        test_utils.layer_test(
            SamePadding,
            kwargs={'kernel_size': 1, 'strides': 1, 'dilation_rate': 1, 'symmetric_pad': False},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 5, 3],
            expected_output_dtype='float32'
        )

        test_utils.layer_test(
            SamePadding,
            kwargs={'kernel_size': 3, 'strides': 1, 'dilation_rate': 1, 'symmetric_pad': True},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 6, 7, 3],
            expected_output_dtype='float32'
        )

    def test_error(self):
        layer = SamePadding(kernel_size=3, strides=1, dilation_rate=5, symmetric_pad=False)
        layer(tf.zeros((1, 3, 4, 1)))

        with self.assertRaisesRegex(ValueError, 'Unable to use symmetric padding'):
            layer = SamePadding(kernel_size=3, strides=1, dilation_rate=5, symmetric_pad=True)


if __name__ == '__main__':
    tf.test.main()

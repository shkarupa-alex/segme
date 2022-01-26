import tensorflow as tf
from keras import keras_parameterized, testing_utils
from keras.mixed_precision import policy as mixed_precision
from ..aspp import AtrousSeparableConv, ASPPPool, ASPP


@keras_parameterized.run_all_keras_modes
class TestAtrousSeparableConv(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            AtrousSeparableConv,
            kwargs={'filters': 10, 'dilation': 1, 'standardized': False},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            AtrousSeparableConv,
            kwargs={'filters': 64, 'dilation': 4, 'standardized': True},
            input_shape=[2, 16, 16, 32],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 64],
            expected_output_dtype='float32'
        )


@keras_parameterized.run_all_keras_modes
class TestASPPPool(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            ASPPPool,
            kwargs={'filters': 10, 'standardized': False},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            ASPPPool,
            kwargs={'filters': 64, 'standardized': True},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 64],
            expected_output_dtype='float32'
        )


@keras_parameterized.run_all_keras_modes
class TestASPP(keras_parameterized.TestCase):
    def setUp(self):
        super(TestASPP, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestASPP, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            ASPP,
            kwargs={'filters': 10, 'stride': 8, 'standardized': False},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            ASPP,
            kwargs={'filters': 64, 'stride': 16, 'standardized': True},
            input_shape=[2, 16, 16, 32],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 64],
            expected_output_dtype='float32'
        )

        mixed_precision.set_global_policy('mixed_float16')
        testing_utils.layer_test(
            ASPP,
            kwargs={'filters': 64, 'stride': 32, 'standardized': True},
            input_shape=[2, 16, 16, 32],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 64],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()

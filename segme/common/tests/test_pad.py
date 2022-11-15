import numpy as np
import tensorflow as tf
from keras import layers
from keras.mixed_precision import policy as mixed_precision
from keras.utils import custom_object_scope
from keras.testing_infra import test_combinations, test_utils
from segme.common.pad import SymmetricPadding, with_divisible_pad


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


class OddConstrainedLayer(layers.Layer):
    def __init__(self, data_format='channels_last', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_format = data_format
        self.data_format_ = 'NHWC' if 'channels_last' == data_format else 'NCHW'

    def constraned_op(self, inputs, **kwargs):
        outputs = tf.nn.space_to_depth(inputs, 2, data_format=self.data_format_)
        outputs -= 1.
        outputs = tf.nn.depth_to_space(outputs, 2, data_format=self.data_format_)

        return outputs

    def call(self, inputs, *args, **kwargs):
        outputs = with_divisible_pad(self.constraned_op, inputs, 2, data_format=self.data_format)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({'data_format': self.data_format})

        return config


@test_combinations.run_all_keras_modes
class TestWithDivisiblePad(test_combinations.TestCase):
    def setUp(self):
        super(TestWithDivisiblePad, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestWithDivisiblePad, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        with custom_object_scope({'OddConstrainedLayer': OddConstrainedLayer}):
            test_utils.layer_test(
                OddConstrainedLayer,
                kwargs={'data_format': 'channels_last'},
                input_shape=[2, 4, 5, 3],
                input_dtype='float32',
                expected_output_shape=[None, 4, 5, 3],
                expected_output_dtype='float32'
            )

            if tf.test.is_gpu_available():
                test_utils.layer_test(
                    OddConstrainedLayer,
                    kwargs={'data_format': 'channels_first'},
                    input_shape=[2, 3, 5, 4],
                    input_dtype='float32',
                    expected_output_shape=[None, 3, 5, 4],
                    expected_output_dtype='float32'
                )

    def test_value(self):
        inputs = np.arange(2 * 3 * 5 * 4).astype('float32').reshape([2, 3, 5, 4])

        result = OddConstrainedLayer()(inputs)
        result = self.evaluate(result)
        self.assertAllClose(result, inputs - 1.)


if __name__ == '__main__':
    tf.test.main()

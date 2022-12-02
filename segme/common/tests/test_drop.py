import numpy as np
import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.common.drop import DropPath, DropBlock


@test_combinations.run_all_keras_modes
class TestDropPath(test_combinations.TestCase):
    def setUp(self):
        super(TestDropPath, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDropPath, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            DropPath,
            kwargs={'rate': 0.},
            input_shape=[2, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DropPath,
            kwargs={'rate': 0.2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            DropPath,
            kwargs={'rate': 0.1},
            input_shape=[2, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 3],
            expected_output_dtype='float16'
        )


@test_combinations.run_all_keras_modes
class TestDropBlock(test_combinations.TestCase):
    def setUp(self):
        super(TestDropBlock, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDropBlock, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            DropBlock,
            kwargs={'rate': 0., 'size': 2},
            input_shape=[2, 16, 3, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 3, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DropBlock,
            kwargs={'rate': 0.2, 'size': 1},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            DropBlock,
            kwargs={'rate': 0.1, 'size': 3},
            input_shape=[2, 16, 3, 1],
            input_dtype='float16',
            expected_output_shape=[None, 16, 3, 1],
            expected_output_dtype='float16'
        )

    def test_mean(self):
        inputs = np.random.uniform(size=[4, 32, 32, 3]).astype('float32')

        outputs = DropBlock(0.2, 3)(inputs, training=True)
        outputs = self.evaluate(outputs)

        self.assertNotAllClose(inputs, outputs)
        self.assertAllClose(inputs.mean(axis=(1, 2)), outputs.mean(axis=(1, 2)), atol=5e-2)


if __name__ == '__main__':
    tf.test.main()

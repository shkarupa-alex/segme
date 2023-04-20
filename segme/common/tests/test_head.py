import tensorflow as tf
from keras import mixed_precision
from keras.src.testing_infra import test_combinations, test_utils
from segme.common.head import HeadProjection, ClassificationActivation, ClassificationHead


@test_combinations.run_all_keras_modes
class TestHeadProjection(test_combinations.TestCase):
    def setUp(self):
        super(TestHeadProjection, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestHeadProjection, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            HeadProjection,
            kwargs={'classes': 2, 'kernel_size': 3},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 2],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            HeadProjection,
            kwargs={'classes': 4, 'kernel_size': 1},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float16'
        )


@test_combinations.run_all_keras_modes
class TestClassificationActivation(test_combinations.TestCase):
    def setUp(self):
        super(TestClassificationActivation, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestClassificationActivation, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            ClassificationActivation,
            kwargs={},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            ClassificationActivation,
            kwargs={},
            input_shape=[2, 16, 16, 1],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 1],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            ClassificationActivation,
            kwargs={'dtype': 'float32'},
            input_shape=[2, 16, 16, 1],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 1],
            expected_output_dtype='float32'
        )


@test_combinations.run_all_keras_modes
class TestClassificationHead(test_combinations.TestCase):
    def setUp(self):
        super(TestClassificationHead, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestClassificationHead, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            ClassificationHead,
            kwargs={'classes': 2, 'kernel_size': 1, 'kernel_initializer': 'variance_scaling'},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 2],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            ClassificationHead,
            kwargs={'classes': 1, 'kernel_size': 3, 'kernel_initializer': 'glorot_uniform'},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 1],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            ClassificationHead,
            kwargs={'classes': 4, 'kernel_size': 1, 'kernel_initializer': 'glorot_uniform'},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            ClassificationHead,
            kwargs={'classes': 4, 'kernel_size': 1, 'kernel_initializer': 'glorot_uniform', 'dtype': 'float32'},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()

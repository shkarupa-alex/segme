import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from ..head import ClassificationHead


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
            kwargs={'classes': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 2],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            ClassificationHead,
            kwargs={'classes': 1},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 1],
            expected_output_dtype='float32'
        )

        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            ClassificationHead,
            kwargs={'classes': 4},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()

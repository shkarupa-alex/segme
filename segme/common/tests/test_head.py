import tensorflow as tf
from keras import keras_parameterized, testing_utils
from keras.mixed_precision import policy as mixed_precision
from ..head import ClassificationHead


@keras_parameterized.run_all_keras_modes
class TestClassificationHead(keras_parameterized.TestCase):
    def setUp(self):
        super(TestClassificationHead, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestClassificationHead, self).tearDown()
        mixed_precision.set_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            ClassificationHead,
            kwargs={'classes': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 2],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            ClassificationHead,
            kwargs={'classes': 1},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 1],
            expected_output_dtype='float32'
        )

        mixed_precision.set_policy('mixed_float16')
        testing_utils.layer_test(
            ClassificationHead,
            kwargs={'classes': 4},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()

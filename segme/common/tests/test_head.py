import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..head import ClassificationHead, RegressionHead


@keras_parameterized.run_all_keras_modes
class TestClassificationHead(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            ClassificationHead,
            kwargs={'classes': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 1],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            ClassificationHead,
            kwargs={'classes': 4},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )


@keras_parameterized.run_all_keras_modes
class TestRegressionHead(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            RegressionHead,
            kwargs={},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 1],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            RegressionHead,
            kwargs={},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 1],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()

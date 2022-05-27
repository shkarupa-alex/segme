import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from ..single import SingleConvBlock


@test_combinations.run_all_keras_modes
class TestSingleConvBlock(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            SingleConvBlock,
            kwargs={'out_features': 10},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()

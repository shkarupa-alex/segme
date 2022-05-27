import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from ..double import DoubleConvBlock


@test_combinations.run_all_keras_modes
class TestDoubleConvBlock(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            DoubleConvBlock,
            kwargs={'mid_features': 10},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 10],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DoubleConvBlock,
            kwargs={'mid_features': 10, 'out_features': 5, 'stride': 2},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 8, 8, 5],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()

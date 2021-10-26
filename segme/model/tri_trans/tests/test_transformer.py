import tensorflow as tf
from keras import keras_parameterized, testing_utils
from ..transformer import DecoderCup, VisionTransformer


@keras_parameterized.run_all_keras_modes
class TestDecoderCup(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            DecoderCup,
            kwargs={'filters': 32},
            input_shape=(2, 49, 16),
            input_dtype='float32',
            expected_output_shape=(None, 7, 7, 32),
            expected_output_dtype='float32'
        )


@keras_parameterized.run_all_keras_modes
class TestVisionTransformer(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            VisionTransformer,
            kwargs={},
            input_shape=(2, 14, 14, 32),
            input_dtype='float32',
            expected_output_shape=(None, 14, 14, 32),
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            VisionTransformer,
            kwargs={},
            input_shape=(2, 18, 18, 14),
            input_dtype='float32',
            expected_output_shape=(None, 18, 18, 14),
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()

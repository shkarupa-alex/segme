import tensorflow as tf
from keras import keras_parameterized, testing_utils
from ..uniatt import UnionAttention


@keras_parameterized.run_all_keras_modes
class TestUnionAttention(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            UnionAttention,
            kwargs={'confidence': 0.1},
            input_shape=(2, 32, 32, 16),
            input_dtype='float32',
            expected_output_shape=(None, 32, 32, 1),
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()

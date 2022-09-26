import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from segme.model.sod.tracer.uniatt import UnionAttention


@test_combinations.run_all_keras_modes
class TestUnionAttention(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            UnionAttention,
            kwargs={'confidence': 0.1},
            input_shape=(2, 32, 32, 16),
            input_dtype='float32',
            expected_output_shape=(None, 32, 32, 1),
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()

import tensorflow as tf
from tf_keras.src.testing_infra import test_combinations
from segme.model.sod.tracer.objatt import ObjectAttention
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestObjectAttention(test_combinations.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            ObjectAttention,
            kwargs={'denoise': 0.93},
            input_shapes=[(2, 32, 32, 16), (2, 32, 32, 1)],
            input_dtypes=['float32'] * 2,
            expected_output_shapes=[(None, 32, 32, 1)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()

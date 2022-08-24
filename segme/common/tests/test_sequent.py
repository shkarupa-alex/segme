import numpy as np
import tensorflow as tf
from keras import layers
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.common.sequent import Sequential
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestSequential(test_combinations.TestCase):
    def setUp(self):
        super(TestSequential, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSequential, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            Sequential,
            kwargs={'items': [layers.BatchNormalization(), layers.ReLU()]},
            input_shape=(2, 16, 16, 10),
            input_dtype='float32',
            expected_output_shape=(None, 16, 16, 10),
            expected_output_dtype='float32'
        )
        layer_multi_io_test(
            Sequential,
            kwargs={'items': [layers.ReLU(), layers.Lambda(lambda x: tf.split(x, 2, axis=-1))]},
            input_shapes=[(2, 16, 16, 10)],
            input_dtypes=['float32'],
            expected_output_shapes=[(None, 16, 16, 5), (None, 16, 16, 5)],
            expected_output_dtypes=['float32'] * 2
        )
        layer_multi_io_test(
            Sequential,
            kwargs={'items': [layers.ReLU(), layers.Lambda(lambda x: tf.split(x, 2, axis=-1)), layers.Add()]},
            input_shapes=[(2, 16, 16, 10)],
            input_dtypes=['float32'],
            expected_output_shapes=[(None, 16, 16, 5)],
            expected_output_dtypes=['float32']
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            Sequential,
            kwargs={'items': [layers.BatchNormalization(), layers.ReLU()]},
            input_shape=(2, 16, 16, 10),
            input_dtype='float16',
            expected_output_shape=(None, 16, 16, 10),
            expected_output_dtype='float16'
        )
        layer_multi_io_test(
            Sequential,
            kwargs={'items': [layers.ReLU(), layers.Lambda(lambda x: tf.split(x, 2, axis=-1))]},
            input_shapes=[(2, 16, 16, 10)],
            input_dtypes=['float16'],
            expected_output_shapes=[(None, 16, 16, 5), (None, 16, 16, 5)],
            expected_output_dtypes=['float16'] * 2
        )
        layer_multi_io_test(
            Sequential,
            kwargs={'items': [layers.ReLU(), layers.Lambda(lambda x: tf.split(x, 2, axis=-1)), layers.Add()]},
            input_shapes=[(2, 16, 16, 10)],
            input_dtypes=['float16'],
            expected_output_shapes=[(None, 16, 16, 5)],
            expected_output_dtypes=['float16']
        )


if __name__ == '__main__':
    tf.test.main()

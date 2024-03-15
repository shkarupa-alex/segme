import tensorflow as tf
from tf_keras import layers, mixed_precision, utils
from tf_keras.src.testing_infra import test_combinations, test_utils
from segme.common.sequence import Sequence
from segme.testing_utils import layer_multi_io_test


class Split2(layers.Layer):
    def call(self, inputs, *args, **kwargs):
        return tf.split(inputs, 2, axis=-1)


@test_combinations.run_all_keras_modes
class TestSequence(test_combinations.TestCase):
    def setUp(self):
        super(TestSequence, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSequence, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        with utils.custom_object_scope({'Split2': Split2}):
            test_utils.layer_test(
                Sequence,
                kwargs={'items': [layers.BatchNormalization(), layers.ReLU()]},
                input_shape=(2, 16, 16, 10),
                input_dtype='float32',
                expected_output_shape=(None, 16, 16, 10),
                expected_output_dtype='float32'
            )
            layer_multi_io_test(
                Sequence,
                kwargs={'items': [layers.ReLU(), Split2()]},
                input_shapes=[(2, 16, 16, 10)],
                input_dtypes=['float32'],
                expected_output_shapes=[(None, 16, 16, 5), (None, 16, 16, 5)],
                expected_output_dtypes=['float32'] * 2
            )
            layer_multi_io_test(
                Sequence,
                kwargs={'items': [layers.ReLU(), Split2(), layers.Add()]},
                input_shapes=[(2, 16, 16, 10)],
                input_dtypes=['float32'],
                expected_output_shapes=[(None, 16, 16, 5)],
                expected_output_dtypes=['float32']
            )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        with utils.custom_object_scope({'Split2': Split2}):
            test_utils.layer_test(
                Sequence,
                kwargs={'items': [layers.BatchNormalization(), layers.ReLU()]},
                input_shape=(2, 16, 16, 10),
                input_dtype='float16',
                expected_output_shape=(None, 16, 16, 10),
                expected_output_dtype='float16'
            )
            test_utils.layer_test(
                Sequence,
                kwargs={'items': [layers.BatchNormalization(), layers.ReLU(dtype='float32')]},
                input_shape=(2, 16, 16, 10),
                input_dtype='float16',
                expected_output_shape=(None, 16, 16, 10),
                expected_output_dtype='float32'
            )
            layer_multi_io_test(
                Sequence,
                kwargs={'items': [layers.ReLU(), Split2()]},
                input_shapes=[(2, 16, 16, 10)],
                input_dtypes=['float16'],
                expected_output_shapes=[(None, 16, 16, 5), (None, 16, 16, 5)],
                expected_output_dtypes=['float16'] * 2
            )
            layer_multi_io_test(
                Sequence,
                kwargs={'items': [layers.ReLU(), Split2(), layers.Add()]},
                input_shapes=[(2, 16, 16, 10)],
                input_dtypes=['float16'],
                expected_output_shapes=[(None, 16, 16, 5)],
                expected_output_dtypes=['float16']
            )


if __name__ == '__main__':
    tf.test.main()

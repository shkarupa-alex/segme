import tensorflow as tf
from tf_keras import mixed_precision
from tf_keras.src.testing_infra import test_combinations, test_utils
from segme.common.split import Split
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestSplit(test_combinations.TestCase):
    def setUp(self):
        super(TestSplit, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSplit, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            Split,
            kwargs={'num_or_size_splits': 2, 'axis': -1},
            input_shapes=[(2, 16, 4)],
            input_dtypes=['float32'],
            expected_output_shapes=[(None, 16, 2)] * 2,
            expected_output_dtypes=['float32'] * 2
        )
        layer_multi_io_test(
            Split,
            kwargs={'num_or_size_splits': [2, 8, 6], 'axis': 1},
            input_shapes=[(2, 16, 4)],
            input_dtypes=['float32'],
            expected_output_shapes=[(None, 2, 4), (None, 8, 4), (None, 6, 4)],
            expected_output_dtypes=['float32'] * 3
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            Split,
            kwargs={'num_or_size_splits': 3, 'axis': 1},
            input_shapes=[(2, 15, 4)],
            input_dtypes=['float16'],
            expected_output_shapes=[(None, 5, 4)] * 3,
            expected_output_dtypes=['float16'] * 3
        )


if __name__ == '__main__':
    tf.test.main()

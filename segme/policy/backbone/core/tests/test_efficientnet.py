import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations
from segme.common.backbone import Backbone
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestEfficientNet(test_combinations.TestCase):
    def setUp(self):
        super(TestEfficientNet, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestEfficientNet, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_small(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'efficientnet_v2_small-imagenet'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 24),
                (None, 56, 56, 48),
                (None, 28, 28, 64),
                (None, 14, 14, 160),
                (None, 7, 7, 1280)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    def test_small_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'efficientnet_v2_small-imagenet'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 24),
                (None, 56, 56, 48),
                (None, 28, 28, 64),
                (None, 14, 14, 160),
                (None, 7, 7, 1280)
            ],
            expected_output_dtypes=['float16'] * 5
        )

    def test_medium(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'efficientnet_v2_medium-none'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 24),
                (None, 56, 56, 48),
                (None, 28, 28, 80),
                (None, 14, 14, 176),
                (None, 7, 7, 1280)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    def test_large(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'efficientnet_v2_large-none'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 32),
                (None, 56, 56, 64),
                (None, 28, 28, 96),
                (None, 14, 14, 224),
                (None, 7, 7, 1280)
            ],
            expected_output_dtypes=['float32'] * 5
        )


if __name__ == '__main__':
    tf.test.main()

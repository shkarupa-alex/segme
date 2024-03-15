import tensorflow as tf
from tf_keras import mixed_precision
from tf_keras.src.testing_infra import test_combinations
from segme.common.backbone import Backbone
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestMobileNet(test_combinations.TestCase):
    def setUp(self):
        super(TestMobileNet, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestMobileNet, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_small(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'mobilenet_v3_small-imagenet'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 16),
                (None, 56, 56, 72),
                (None, 28, 28, 96),
                (None, 14, 14, 288),
                (None, 7, 7, 576)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    def test_small_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'mobilenet_v3_small-imagenet'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 16),
                (None, 56, 56, 72),
                (None, 28, 28, 96),
                (None, 14, 14, 288),
                (None, 7, 7, 576)
            ],
            expected_output_dtypes=['float16'] * 5
        )

    def test_large(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'mobilenet_v3_large-none'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 64),
                (None, 56, 56, 72),
                (None, 28, 28, 240),
                (None, 14, 14, 672),
                (None, 7, 7, 960)
            ],
            expected_output_dtypes=['float32'] * 5
        )


if __name__ == '__main__':
    tf.test.main()

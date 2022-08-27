import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations
from segme.common.backbone import Backbone
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestGCViT(test_combinations.TestCase):
    def setUp(self):
        super(TestGCViT, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestGCViT, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_nano(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'gcvit_nano-imagenet'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 64),
                (None, 56, 56, 64),
                (None, 28, 28, 128),
                (None, 14, 14, 256),
                (None, 7, 7, 512)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    def test_nano_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'gcvit_nano-imagenet'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 64),
                (None, 56, 56, 64),
                (None, 28, 28, 128),
                (None, 14, 14, 256),
                (None, 7, 7, 512)
            ],
            expected_output_dtypes=['float16'] * 5
        )

    def test_micro(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'gcvit_micro-none'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 64),
                (None, 56, 56, 64),
                (None, 28, 28, 128),
                (None, 14, 14, 256),
                (None, 7, 7, 512)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    def test_tiny(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'gcvit_tiny-none'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 64),
                (None, 56, 56, 64),
                (None, 28, 28, 128),
                (None, 14, 14, 256),
                (None, 7, 7, 512)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    def test_small(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'gcvit_small-none'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 96),
                (None, 56, 56, 96),
                (None, 28, 28, 192),
                (None, 14, 14, 384),
                (None, 7, 7, 768)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    def test_base(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'gcvit_base-none'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 128),
                (None, 56, 56, 128),
                (None, 28, 28, 256),
                (None, 14, 14, 512),
                (None, 7, 7, 1024)
            ],
            expected_output_dtypes=['float32'] * 5
        )


if __name__ == '__main__':
    tf.test.main()

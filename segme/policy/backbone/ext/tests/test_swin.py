import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations
from segme.common.backbone import Backbone
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestSwinV1(test_combinations.TestCase):
    def setUp(self):
        super(TestSwinV1, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSwinV1, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_tiny_224(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'swin_tiny_224-imagenet'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 56, 56, 96),
                (None, 28, 28, 192),
                (None, 14, 14, 384),
                (None, 7, 7, 768)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    def test_tiny_224_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'swin_tiny_224-imagenet'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 56, 56, 96),
                (None, 28, 28, 192),
                (None, 14, 14, 384),
                (None, 7, 7, 768)
            ],
            expected_output_dtypes=['float16'] * 4
        )

    def test_small_224(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'swin_small_224-none'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 56, 56, 96),
                (None, 28, 28, 192),
                (None, 14, 14, 384),
                (None, 7, 7, 768)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    def test_base_224(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'swin_base_224-none'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 56, 56, 128),
                (None, 28, 28, 256),
                (None, 14, 14, 512),
                (None, 7, 7, 1024)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    def test_base_384(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'swin_base_384-none'},
            input_shapes=[(2, 384, 384, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 96, 96, 128),
                (None, 48, 48, 256),
                (None, 24, 24, 512),
                (None, 12, 12, 1024)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    def test_large_224(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'swin_large_224-none'},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 56, 56, 192),
                (None, 28, 28, 384),
                (None, 14, 14, 768),
                (None, 7, 7, 1536)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    def test_large_384(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'swin_large_384-none'},
            input_shapes=[(2, 384, 384, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 96, 96, 192),
                (None, 48, 48, 384),
                (None, 24, 24, 768),
                (None, 12, 12, 1536)
            ],
            expected_output_dtypes=['float32'] * 4
        )


@test_combinations.run_all_keras_modes
class TestSwinV2(test_combinations.TestCase):
    def setUp(self):
        super(TestSwinV2, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSwinV2, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_tiny_256(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'swin_v2_tiny_256-imagenet'},
            input_shapes=[(2, 256, 256, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 64, 64, 96),
                (None, 32, 32, 192),
                (None, 16, 16, 384),
                (None, 8, 8, 768)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    def test_tiny_256_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'swin_v2_tiny_256-imagenet'},
            input_shapes=[(2, 256, 256, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 64, 64, 96),
                (None, 32, 32, 192),
                (None, 16, 16, 384),
                (None, 8, 8, 768)
            ],
            expected_output_dtypes=['float16'] * 4
        )

    def test_small_256(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'swin_v2_small_256-none'},
            input_shapes=[(2, 256, 256, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 64, 64, 96),
                (None, 32, 32, 192),
                (None, 16, 16, 384),
                (None, 8, 8, 768)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    def test_base_256(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'swin_v2_base_256-none'},
            input_shapes=[(2, 256, 256, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 64, 64, 128),
                (None, 32, 32, 256),
                (None, 16, 16, 512),
                (None, 8, 8, 1024)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    def test_base_384(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'swin_v2_base_384-none'},
            input_shapes=[(2, 384, 384, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 96, 96, 128),
                (None, 48, 48, 256),
                (None, 24, 24, 512),
                (None, 12, 12, 1024)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    def test_large_256(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'swin_v2_large_256-none'},
            input_shapes=[(2, 256, 256, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 64, 64, 192),
                (None, 32, 32, 384),
                (None, 16, 16, 768),
                (None, 8, 8, 1536)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    def test_large_384(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'scales': None, 'policy': 'swin_v2_large_384-none'},
            input_shapes=[(2, 384, 384, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 96, 96, 192),
                (None, 48, 48, 384),
                (None, 24, 24, 768),
                (None, 12, 12, 1536)
            ],
            expected_output_dtypes=['float32'] * 4
        )


if __name__ == '__main__':
    tf.test.main()

import tensorflow as tf
from absl.testing import parameterized
from keras import keras_parameterized
from ..backbone import Backbone
from ...testing_utils import layer_multi_io_test

_CUSTOM_TESTS = {
    'inception_v3', 'inception_resnet_v2', 'xception', 'vgg_16', 'vgg_19',
    'swin_tiny_224', 'swin_small_224', 'swin_base_224', 'swin_base_384', 'swin_large_224', 'swin_large_384',
    'van_tiny', 'van_small', 'van_base', 'van_large',
    'aligned_xception_41_stride_16', 'aligned_xception_65_stride_16', 'aligned_xception_71_stride_16',
    'aligned_xception_41_stride_8', 'aligned_xception_65_stride_8', 'aligned_xception_71_stride_8',
    'bit_m_r50x1_stride_8'}
_DEFAULT_TEST = set(Backbone._config.keys()) - _CUSTOM_TESTS
_DEFAULT_IMAGENET_TEST = _DEFAULT_TEST - {'aligned_xception_41', 'aligned_xception_65', 'aligned_xception_71'}


@keras_parameterized.run_all_keras_modes
class TestBackbone(keras_parameterized.TestCase):
    def test_arch_config(self):
        archs = Backbone._config
        for arch_name in archs:
            arch = archs[arch_name]
            self.assertIsInstance(arch, tuple)
            self.assertLen(arch, 2)

            model, feats = arch
            self.assertTrue(callable(model))
            self.assertIsInstance(feats, tuple)
            self.assertLen(feats, 6)

            for ft in feats:
                self.assertIsInstance(ft, (type(None), str, int))

    @parameterized.parameters(_DEFAULT_TEST)
    def test_layer_default_trainable(self, arch_name):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': arch_name, 'init': None, 'trainable': True},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, None),
                (None, 56, 56, None),
                (None, 28, 28, None),
                (None, 14, 14, None),
                (None, 7, 7, None)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    @parameterized.parameters(_DEFAULT_IMAGENET_TEST)
    def test_layer_default_imagenet(self, arch_name):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': arch_name, 'init': 'imagenet', 'trainable': False},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, None),
                (None, 56, 56, None),
                (None, 28, 28, None),
                (None, 14, 14, None),
                (None, 7, 7, None)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    def test_layer_inception_v3_trainable(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': 'inception_v3', 'init': None, 'trainable': True},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 109, 109, None),
                (None, 52, 52, None),
                (None, 25, 25, None),
                (None, 12, 12, None),
                (None, 5, 5, None)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    def test_layer_inception_v3_imagenet(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': 'inception_v3', 'init': 'imagenet',
                    'trainable': False},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 109, 109, None),
                (None, 52, 52, None),
                (None, 25, 25, None),
                (None, 12, 12, None),
                (None, 5, 5, None)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    def test_layer_inception_resnet_v2_trainable(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': 'inception_resnet_v2', 'init': None,
                    'trainable': True, 'scales': [2, 4]},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 109, 109, None),
                (None, 52, 52, None),
                # (None, 25, 25, None),
                # (None, 12, 12, None),
                # (None, 5, 5, None)
            ],
            expected_output_dtypes=['float32'] * 2
        )

    def test_layer_inception_resnet_v2_imagenet(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': 'inception_resnet_v2', 'init': 'imagenet',
                    'trainable': False, 'scales': [2, 4]},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 109, 109, None),
                (None, 52, 52, None),
                # (None, 25, 25, None),
                # (None, 12, 12, None),
                # (None, 5, 5, None)
            ],
            expected_output_dtypes=['float32'] * 2
        )

    def test_layer_xception_trainable(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': 'xception', 'init': None, 'trainable': True},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 109, 109, None),
                (None, 55, 55, None),
                (None, 28, 28, None),
                (None, 14, 14, None),
                (None, 7, 7, None)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    def test_layer_xception_imagenet(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': 'xception', 'init': 'imagenet',
                    'trainable': False},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 109, 109, None),
                (None, 55, 55, None),
                (None, 28, 28, None),
                (None, 14, 14, None),
                (None, 7, 7, None)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    @parameterized.parameters(['vgg_16', 'vgg_19'])
    def test_layer_vgg_trainable(self, vgg_arch):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': vgg_arch, 'init': None, 'trainable': True},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 224, 224, None),
                (None, 112, 112, None),
                (None, 56, 56, None),
                (None, 28, 28, None),
                (None, 14, 14, None)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    @parameterized.parameters(['vgg_16', 'vgg_19'])
    def test_layer_vgg_imagenet(self, vgg_arch):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': vgg_arch, 'init': 'imagenet',
                    'trainable': False},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 224, 224, None),
                (None, 112, 112, None),
                (None, 56, 56, None),
                (None, 28, 28, None),
                (None, 14, 14, None)
            ],
            expected_output_dtypes=['float32'] * 5
        )

    @parameterized.parameters(['swin_tiny_224', 'swin_small_224', 'swin_base_224', 'swin_large_224'])
    def test_layer_swin_trainable_224(self, swin_arch):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': swin_arch, 'init': None, 'trainable': True},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 56, 56, None),
                (None, 28, 28, None),
                (None, 14, 14, None),
                (None, 7, 7, None)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    @parameterized.parameters(['swin_base_384', 'swin_large_384'])
    def test_layer_swin_trainable_384(self, swin_arch):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': swin_arch, 'init': None, 'trainable': True},
            input_shapes=[(2, 384, 384, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 96, 96, None),
                (None, 48, 48, None),
                (None, 24, 24, None),
                (None, 12, 12, None)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    @parameterized.parameters(['swin_tiny_224', 'swin_small_224', 'swin_base_224', 'swin_large_224'])
    def test_layer_swin_imagenet_224(self, swin_arch):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': swin_arch, 'init': 'imagenet',
                    'trainable': False},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 56, 56, None),
                (None, 28, 28, None),
                (None, 14, 14, None),
                (None, 7, 7, None)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    @parameterized.parameters(['swin_base_384', 'swin_large_384'])
    def test_layer_swin_imagenet_384(self, swin_arch):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': swin_arch, 'init': 'imagenet',
                    'trainable': False},
            input_shapes=[(2, 384, 384, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 96, 96, None),
                (None, 48, 48, None),
                (None, 24, 24, None),
                (None, 12, 12, None)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    @parameterized.parameters(['van_tiny', 'van_small', 'van_base', 'van_large'])
    def test_layer_van_trainable(self, van_arch):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': van_arch, 'init': None, 'trainable': True},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 56, 56, None),
                (None, 28, 28, None),
                (None, 14, 14, None),
                (None, 7, 7, None)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    @parameterized.parameters(['van_tiny', 'van_small', 'van_base', 'van_large'])
    def test_layer_van_imagenet(self, van_arch):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': van_arch, 'init': 'imagenet',
                    'trainable': False},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 56, 56, None),
                (None, 28, 28, None),
                (None, 14, 14, None),
                (None, 7, 7, None)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    @parameterized.parameters([
        'aligned_xception_41_stride_16', 'aligned_xception_65_stride_16', 'aligned_xception_71_stride_16'])
    def test_layer_aligned_xception_stride_16_trainable(self, ae_arch):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': ae_arch, 'init': None, 'trainable': True},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, None),
                (None, 56, 56, None),
                (None, 28, 28, None),
                (None, 14, 14, None)
            ],
            expected_output_dtypes=['float32'] * 4
        )

    @parameterized.parameters([
        'aligned_xception_41_stride_8', 'aligned_xception_65_stride_8', 'aligned_xception_71_stride_8'])
    def test_layer_aligned_xception_stride_8_trainable(self, ae_arch):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': ae_arch, 'init': None, 'trainable': True},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, None),
                (None, 56, 56, None),
                (None, 28, 28, None)
            ],
            expected_output_dtypes=['float32'] * 3
        )

    @parameterized.parameters(['bit_m_r50x1_stride_8'])
    def test_layer_bit_stride_8_trainable(self, bit_arch):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': bit_arch, 'init': None, 'trainable': True},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 64),
                (None, 56, 56, 256),
                (None, 28, 28, 2048)
            ],
            expected_output_dtypes=['float32'] * 3
        )

    @parameterized.parameters(['bit_m_r50x1_stride_8'])
    def test_layer_bit_stride_8_imagenet(self, bit_arch):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': bit_arch, 'init': 'imagenet', 'trainable': False},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, 64),
                (None, 56, 56, 256),
                (None, 28, 28, 2048)
            ],
            expected_output_dtypes=['float32'] * 3
        )


if __name__ == '__main__':
    tf.test.main()

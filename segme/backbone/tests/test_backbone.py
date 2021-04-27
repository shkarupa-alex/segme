import tensorflow as tf
from absl.testing import parameterized
from tensorflow.python.keras import keras_parameterized
from ..backbone import Backbone
from ...testing_utils import layer_multi_io_test

_CUSTOM_TESTS = {
    'inception_v3', 'inception_resnet_v2', 'xception', 'vgg_16', 'vgg_19',
    'aligned_xception_41_stride_16', 'aligned_xception_65_stride_16', 'aligned_xception_71_stride_16',
    'aligned_xception_41_stride_8', 'aligned_xception_65_stride_8', 'aligned_xception_71_stride_8'}
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

    def test_layer_aligned_xception_stride_16_trainable(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': 'aligned_xception_41_stride_16', 'init': None, 'trainable': True},
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
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': 'aligned_xception_65_stride_16', 'init': None, 'trainable': True},
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
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': 'aligned_xception_71_stride_16', 'init': None, 'trainable': True},
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

    def test_layer_aligned_xception_stride_8_trainable(self):
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': 'aligned_xception_41_stride_8', 'init': None, 'trainable': True},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, None),
                (None, 56, 56, None),
                (None, 28, 28, None)
            ],
            expected_output_dtypes=['float32'] * 3
        )
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': 'aligned_xception_65_stride_8', 'init': None, 'trainable': True},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, None),
                (None, 56, 56, None),
                (None, 28, 28, None)
            ],
            expected_output_dtypes=['float32'] * 3
        )
        layer_multi_io_test(
            Backbone,
            kwargs={'arch': 'aligned_xception_71_stride_8', 'init': None, 'trainable': True},
            input_shapes=[(2, 224, 224, 3)],
            input_dtypes=['uint8'],
            expected_output_shapes=[
                (None, 112, 112, None),
                (None, 56, 56, None),
                (None, 28, 28, None)
            ],
            expected_output_dtypes=['float32'] * 3
        )


if __name__ == '__main__':
    tf.test.main()

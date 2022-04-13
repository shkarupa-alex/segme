import cv2
import numpy as np
import os
import tensorflow as tf
from keras import keras_parameterized, testing_utils
from ..guidedup import BoxFilter, GuidedFilter, ConvGuidedFilter
from ...testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestBoxFilter(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            BoxFilter,
            kwargs={'radius': 3},
            input_shape=[2, 8, 9, 1],
            input_dtype='float32',
            expected_output_shape=[None, 8, 9, 1],
            expected_output_dtype='float32'
        )

    def test_value(self):
        expected = np.array([
            [256, 330, 408, 490, 518, 546, 480, 410, 336], [410, 525, 645, 770, 805, 840, 735, 625, 510],
            [600, 765, 936, 1113, 1155, 1197, 1044, 885, 720], [826, 1050, 1281, 1519, 1568, 1617, 1407, 1190, 966],
            [1078, 1365, 1659, 1960, 2009, 2058, 1785, 1505, 1218],
            [1032, 1305, 1584, 1869, 1911, 1953, 1692, 1425, 1152],
            [950, 1200, 1455, 1715, 1750, 1785, 1545, 1300, 1050],
            [832, 1050, 1272, 1498, 1526, 1554, 1344, 1130, 912]], 'int32')[..., None]

        result = BoxFilter(3)(np.arange(1, 73).reshape((1, 1, 8, 9)).transpose((0, 2, 3, 1)))
        result = self.evaluate(result[0])

        self.assertAlmostEqual(result.mean(), 1137.6, places=1)
        self.assertAlmostEqual(result.std(), 475.2, places=1)

        self.assertListEqual(expected.tolist(), result.tolist())

    def test_image(self):
        image = os.path.join(os.path.dirname(__file__), 'data', 'guidedup_rgb.jpeg')
        image = cv2.imread(image).astype('float32') / 255.

        result = BoxFilter(64)(image[None])
        result = self.evaluate(result[0])

        self.assertAlmostEqual(result[..., 0].mean(), 6203.0, places=0)
        self.assertAlmostEqual(result[..., 0].std(), 2772.3, places=1)
        self.assertAlmostEqual(result[..., 1].mean(), 7536.0, places=1)
        self.assertAlmostEqual(result[..., 1].std(), 2117.0, places=0)
        self.assertAlmostEqual(result[..., 2].mean(), 10305.0, places=0)
        self.assertAlmostEqual(result[..., 2].std(), 2206.4, places=1)


@keras_parameterized.run_all_keras_modes
class TestGuidedFilter(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            GuidedFilter,
            kwargs={'radius': 3, 'filters': 16, 'kernel_size': 3, 'normalize': False, 'activation': 'relu',
                    'epsilon': 1e-8, 'standardized': False},
            input_shapes=[(2, 8, 9, 1), (2, 8, 9, 1)],
            input_dtypes=['uint8', 'float32'],
            expected_output_shapes=[(None, 8, 9, 1)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            GuidedFilter,
            kwargs={'radius': 3, 'filters': 64, 'kernel_size': 1, 'normalize': True, 'activation': 'relu',
                    'epsilon': 1e-8, 'standardized': True},
            input_shapes=[(2, 8, 9, 3), (2, 8, 9, 1)],
            input_dtypes=['uint8', 'float32'],
            expected_output_shapes=[(None, 8, 9, 1)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            GuidedFilter,
            kwargs={'radius': 3, 'filters': 8, 'kernel_size': 1, 'normalize': True, 'activation': 'leaky_relu',
                    'epsilon': 1e-2, 'standardized': False},
            input_shapes=[(2, 16, 18, 1), (2, 8, 9, 3)],
            input_dtypes=['uint8', 'float32'],
            expected_output_shapes=[(None, 16, 18, 3)],
            expected_output_dtypes=['float32']
        )

    # Original test for GuidedFilter without internal image-to-guidance projection
    # def test_image(self):
    #     image = os.path.join(os.path.dirname(__file__), 'data', 'guidedup_rgb.jpeg')
    #     image = cv2.imread(image).astype('float32') / 255.
    #
    #     ground = os.path.join(os.path.dirname(__file__), 'data', 'guidedup_gt.jpeg')
    #     ground = cv2.imread(ground).astype('float32') / 255.
    #
    #     expected = os.path.join(os.path.dirname(__file__), 'data', 'guidedup_ex.jpeg')
    #     expected = cv2.imread(expected).astype('float32') / 255.
    #
    #     result = GuidedFilter(8)([image[None], ground[None]])
    #     result = self.evaluate(result[0])
    #     result = np.clip(result, 0., 1.)
    #
    #     self.assertLess(np.abs(result - expected).mean(), 0.012)
    #     self.assertLess(np.abs(result - expected).max(), 0.209)


@keras_parameterized.run_all_keras_modes
class TestConvGuidedFilter(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            ConvGuidedFilter,
            kwargs={'radius': 1, 'filters': 16, 'kernel_size': 3, 'normalize': True, 'activation': 'leaky_relu',
                    'standardized': False},
            input_shapes=[(2, 16, 18, 1), (2, 8, 9, 1)],
            input_dtypes=['uint8', 'float32'],
            expected_output_shapes=[(None, 16, 18, 1)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            ConvGuidedFilter,
            kwargs={'radius': 2, 'filters': 64, 'kernel_size': 1, 'normalize': False, 'activation': 'relu',
                    'standardized': False},
            input_shapes=[(2, 24, 27, 3), (2, 8, 9, 1)],
            input_dtypes=['uint8', 'float32'],
            expected_output_shapes=[(None, 24, 27, 1)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            ConvGuidedFilter,
            kwargs={'radius': 3, 'filters': 64, 'kernel_size': 3, 'normalize': True, 'activation': 'leaky_relu',
                    'standardized': True},
            input_shapes=[(2, 16, 18, 1), (2, 8, 9, 3)],
            input_dtypes=['uint8', 'float32'],
            expected_output_shapes=[(None, 16, 18, 3)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()

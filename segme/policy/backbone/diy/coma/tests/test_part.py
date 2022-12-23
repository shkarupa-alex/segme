import numpy as np
import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from tensorflow.python.framework import test_util
from segme.policy.backbone.diy.coma.part import _PARTITION_TYPES, partition_apply, with_partition


@test_util.run_all_in_graph_and_eager_modes
class TestWithPartition(tf.test.TestCase):
    def setUp(self):
        super(TestWithPartition, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestWithPartition, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_window(self):
        inputs = np.arange(1 * 8 * 12 * 1, dtype='float32').reshape([1, 8, 12, 1])
        height, width = inputs.shape[1:3]
        size, dilation_rate = 4, 1

        expected = np.array([
            0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 4, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40,
            41, 42, 43, 8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 48, 49, 50, 51, 60, 61, 62, 63,
            72, 73, 74, 75, 84, 85, 86, 87, 52, 53, 54, 55, 64, 65, 66, 67, 76, 77, 78, 79, 88, 89, 90, 91, 56, 57, 58,
            59, 68, 69, 70, 71, 80, 81, 82, 83, 92, 93, 94, 95], 'int64').reshape([6, 16, 1])

        result = partition_apply(inputs, height, width, 'window_size', size, dilation_rate)
        result = self.evaluate(result)

        self.assertAllEqual(expected, result)

    def test_dilation(self):
        inputs = np.arange(1 * 8 * 24 * 1, dtype='float32').reshape([1, 8, 24, 1])
        height, width = inputs.shape[1:3]
        size, dilation_rate = 4, 2

        expected = np.array([
            0, 2, 4, 6, 48, 50, 52, 54, 96, 98, 100, 102, 144, 146, 148, 150, 1, 3, 5, 7, 49, 51, 53, 55, 97, 99, 101,
            103, 145, 147, 149, 151, 8, 10, 12, 14, 56, 58, 60, 62, 104, 106, 108, 110, 152, 154, 156, 158, 9, 11, 13,
            15, 57, 59, 61, 63, 105, 107, 109, 111, 153, 155, 157, 159, 16, 18, 20, 22, 64, 66, 68, 70, 112, 114, 116,
            118, 160, 162, 164, 166, 17, 19, 21, 23, 65, 67, 69, 71, 113, 115, 117, 119, 161, 163, 165, 167, 24, 26, 28,
            30, 72, 74, 76, 78, 120, 122, 124, 126, 168, 170, 172, 174, 25, 27, 29, 31, 73, 75, 77, 79, 121, 123, 125,
            127, 169, 171, 173, 175, 32, 34, 36, 38, 80, 82, 84, 86, 128, 130, 132, 134, 176, 178, 180, 182, 33, 35, 37,
            39, 81, 83, 85, 87, 129, 131, 133, 135, 177, 179, 181, 183, 40, 42, 44, 46, 88, 90, 92, 94, 136, 138, 140,
            142, 184, 186, 188, 190, 41, 43, 45, 47, 89, 91, 93, 95, 137, 139, 141, 143, 185, 187, 189, 191],
            'int64').reshape([12, 16, 1])

        result = partition_apply(inputs, height, width, 'window_size', size, dilation_rate)
        result = self.evaluate(result)

        self.assertAllEqual(expected, result)

    def test_grid(self):
        inputs = np.arange(1 * 8 * 12 * 1, dtype='float32').reshape([1, 8, 12, 1])
        height, width = inputs.shape[1:3]
        size, dilation_rate = 4, 1

        expected = np.array([
            0, 3, 6, 9, 24, 27, 30, 33, 48, 51, 54, 57, 72, 75, 78, 81, 1, 4, 7, 10, 25, 28, 31, 34, 49, 52, 55, 58, 73,
            76, 79, 82, 2, 5, 8, 11, 26, 29, 32, 35, 50, 53, 56, 59, 74, 77, 80, 83, 12, 15, 18, 21, 36, 39, 42, 45, 60,
            63, 66, 69, 84, 87, 90, 93, 13, 16, 19, 22, 37, 40, 43, 46, 61, 64, 67, 70, 85, 88, 91, 94, 14, 17, 20, 23,
            38, 41, 44, 47, 62, 65, 68, 71, 86, 89, 92, 95], 'int64').reshape([6, 16, 1])

        result = partition_apply(inputs, height, width, 'grid_size', size, dilation_rate)
        result = self.evaluate(result)

        self.assertAllEqual(expected, result)

    def test_inverse(self):
        inputs = np.arange(3 * 24 * 36 * 3, dtype='float32').reshape([3, 24, 36, 3])
        size = 4

        for part_type in _PARTITION_TYPES:
            for dilation_rate in [1, 2, 3]:
                result = with_partition(lambda x, **kwargs: x, inputs, part_type, size, dilation_rate)
                result = self.evaluate(result)
                self.assertAllEqual(inputs, result)

    def test_pad(self):
        inputs = np.arange(3 * 23 * 37 * 3, dtype='float32').reshape([3, 23, 37, 3])
        size = 4

        for part_type in _PARTITION_TYPES:
            for dilation_rate in [1, 2, 3]:
                result = with_partition(lambda x, **kwargs: x + 1., inputs, part_type, size, dilation_rate)
                result = self.evaluate(result)
                self.assertAllEqual(inputs + 1., result)

    def test_channel(self):
        inputs = np.arange(3 * 24 * 36 * 3, dtype='float32').reshape([3, 24, 36, 3])
        size = 4

        for part_type in _PARTITION_TYPES:
            for dilation_rate in [1, 2, 3]:
                result = with_partition(
                    lambda x, **kwargs: tf.reduce_mean(x, axis=-1, keepdims=True), inputs, part_type, size,
                    dilation_rate)
                result = self.evaluate(result)
                self.assertAllEqual(inputs.mean(-1, keepdims=True), result)

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        inputs = np.arange(3 * 24 * 36 * 3, dtype='float16').reshape([3, 24, 36, 3])
        size = 4

        for part_type in _PARTITION_TYPES:
            for dilation_rate in [1, 2, 3]:
                result = with_partition(lambda x, **kwargs: x, inputs, part_type, size, dilation_rate)
                result = self.evaluate(result)
                self.assertAllEqual(inputs, result)


if __name__ == '__main__':
    tf.test.main()

import numpy as np
import tensorflow as tf
from keras import mixed_precision
from tensorflow.python.framework import test_util
from segme.common.part import _PARTITION_TYPES, partition_apply, partition_reverse, with_partition
from segme.common.part import partition_apply_fused, partition_reverse_fused, with_partition_fused
from segme.common.part import halo_partition, halo_partition_fused


@test_util.run_all_in_graph_and_eager_modes
class TestPartitionApply(tf.test.TestCase):
    def setUp(self):
        super(TestPartitionApply, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestPartitionApply, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_window(self):
        inputs = np.arange(1 * 8 * 12 * 1, dtype='float32').reshape([1, 8, 12, 1])
        height, width = inputs.shape[1:3]
        size, dilation_rate = 4, 1

        expected = np.array([
            0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 4, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40,
            41, 42, 43, 8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 48, 49, 50, 51, 60, 61, 62, 63,
            72, 73, 74, 75, 84, 85, 86, 87, 52, 53, 54, 55, 64, 65, 66, 67, 76, 77, 78, 79, 88, 89, 90, 91, 56, 57, 58,
            59, 68, 69, 70, 71, 80, 81, 82, 83, 92, 93, 94, 95], 'int64').reshape([1, 6, 16, 1])

        result = partition_apply(inputs, height, width, 'window_size', size, dilation_rate)
        result = self.evaluate(result)

        self.assertAllEqual(expected, result)

    def test_dilation(self):
        inputs = np.arange(1 * 8 * 16 * 1, dtype='float32').reshape([1, 8, 16, 1])
        height, width = inputs.shape[1:3]
        size, dilation_rate = 4, 2

        expected = np.array([
            0, 2, 4, 6, 32, 34, 36, 38, 64, 66, 68, 70, 96, 98, 100, 102, 1, 3, 5, 7, 33, 35, 37, 39, 65, 67, 69, 71,
            97, 99, 101, 103, 8, 10, 12, 14, 40, 42, 44, 46, 72, 74, 76, 78, 104, 106, 108, 110, 9, 11, 13, 15, 41, 43,
            45, 47, 73, 75, 77, 79, 105, 107, 109, 111, 16, 18, 20, 22, 48, 50, 52, 54, 80, 82, 84, 86, 112, 114, 116,
            118, 17, 19, 21, 23, 49, 51, 53, 55, 81, 83, 85, 87, 113, 115, 117, 119, 24, 26, 28, 30, 56, 58, 60, 62, 88,
            90, 92, 94, 120, 122, 124, 126, 25, 27, 29, 31, 57, 59, 61, 63, 89, 91, 93, 95, 121, 123, 125, 127],
            'int64').reshape([1, 8, 16, 1])

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
            38, 41, 44, 47, 62, 65, 68, 71, 86, 89, 92, 95], 'int64').reshape([1, 6, 16, 1])

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


@test_util.run_all_in_graph_and_eager_modes
class TestPartitionApplyFused(tf.test.TestCase):
    def setUp(self):
        super(TestPartitionApplyFused, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestPartitionApplyFused, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_apply(self):
        inputs = np.arange(3 * 24 * 48 * 4 * 3 * 5, dtype='float32').reshape([3, 24, 48, 4 * 3 * 5])
        height, width = inputs.shape[1:3]
        size, channels = 4, 20

        for part_type in _PARTITION_TYPES:
            for dilation_rate in [1, 2, 3]:
                for num_heads in [1, 2, 4]:
                    expected = partition_apply(inputs, height, width, part_type, size, dilation_rate)
                    expected = tf.reshape(expected, expected.shape[:-1] + (num_heads, 3 * channels // num_heads))
                    expected = tf.transpose(expected, [0, 1, 3, 2, 4])
                    expected = self.evaluate(expected)

                    result = partition_apply_fused(inputs, height, width, part_type, size, num_heads, dilation_rate)
                    result = self.evaluate(result)

                    self.assertAllEqual(expected, result)

    def test_reverse(self):
        inputs = np.arange(216 * 4 * 16 * 5, dtype='float32')
        height, width = 24, 48
        size, channels = 4, 20

        for part_type in _PARTITION_TYPES:
            for dilation_rate in [1, 2, 3]:
                win_size = size ** 2
                num_wind = height * width // (size * dilation_rate) ** 2
                if part_type not in {'window_size', 'grid_size'}:
                    win_size, num_wind = num_wind, win_size

                for num_heads in [1, 2, 4]:
                    expected = inputs.reshape([-1, num_wind, num_heads, win_size, channels // num_heads])
                    expected = tf.transpose(expected, perm=[0, 1, 3, 4, 2])
                    expected = tf.reshape(expected, [-1, num_wind, win_size, channels])
                    expected = partition_reverse(expected, height, width, part_type, size, dilation_rate)
                    expected = self.evaluate(expected)

                    result = inputs.reshape([-1, num_wind, num_heads, win_size, channels // num_heads])
                    result = partition_reverse_fused(result, height, width, part_type, size, num_heads, dilation_rate)
                    result = self.evaluate(result)

                    self.assertAllEqual(expected, result)

    def test_window(self):
        inputs = np.arange(1 * 8 * 12 * 1, dtype='float32').reshape([1, 8, 12, 1])
        height, width = inputs.shape[1:3]
        size, dilation_rate = 4, 1

        expected = np.array([
            0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 4, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40,
            41, 42, 43, 8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 48, 49, 50, 51, 60, 61, 62, 63,
            72, 73, 74, 75, 84, 85, 86, 87, 52, 53, 54, 55, 64, 65, 66, 67, 76, 77, 78, 79, 88, 89, 90, 91, 56, 57, 58,
            59, 68, 69, 70, 71, 80, 81, 82, 83, 92, 93, 94, 95], 'int64').reshape([1, 6, 1, 16, 1])

        result = partition_apply_fused(inputs, height, width, 'window_size', size, 1, dilation_rate)
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
            'int64').reshape([1, 12, 1, 16, 1])

        result = partition_apply_fused(inputs, height, width, 'window_size', size, 1, dilation_rate)
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
            38, 41, 44, 47, 62, 65, 68, 71, 86, 89, 92, 95], 'int64').reshape([1, 6, 1, 16, 1])

        result = partition_apply_fused(inputs, height, width, 'grid_size', size, 1, dilation_rate)
        result = self.evaluate(result)

        self.assertAllEqual(expected, result)

    def test_inverse(self):
        inputs = np.arange(3 * 24 * 36 * 3, dtype='float32').reshape([3, 24, 36, 3])
        size = 4

        for part_type in _PARTITION_TYPES:
            for dilation_rate in [1, 2, 3]:
                result = with_partition_fused(lambda x, **kwargs: x, inputs, part_type, size, 1, dilation_rate)
                result = self.evaluate(result)
                self.assertAllEqual(inputs, result)

    def test_pad(self):
        inputs = np.arange(3 * 23 * 37 * 3, dtype='float32').reshape([3, 23, 37, 3])
        size = 4

        for part_type in _PARTITION_TYPES:
            for dilation_rate in [1, 2, 3]:
                result = with_partition_fused(lambda x, **kwargs: x + 1., inputs, part_type, size, 1, dilation_rate)
                result = self.evaluate(result)
                self.assertAllEqual(inputs + 1., result)

    def test_channel(self):
        inputs = np.arange(3 * 24 * 36 * 3, dtype='float32').reshape([3, 24, 36, 3])
        size = 4

        for part_type in _PARTITION_TYPES:
            for dilation_rate in [1, 2, 3]:
                result = with_partition_fused(
                    lambda x, **kwargs: tf.reduce_mean(x, axis=-1, keepdims=True),
                    inputs, part_type, size, 1, dilation_rate)
                result = self.evaluate(result)
                self.assertAllEqual(inputs.mean(-1, keepdims=True), result)

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        inputs = np.arange(3 * 24 * 36 * 3, dtype='float16').reshape([3, 24, 36, 3])
        size = 4

        for part_type in _PARTITION_TYPES:
            for dilation_rate in [1, 2, 3]:
                result = with_partition_fused(lambda x, **kwargs: x, inputs, part_type, size, 1, dilation_rate)
                result = self.evaluate(result)
                self.assertAllEqual(inputs, result)


@test_util.run_all_in_graph_and_eager_modes
class TestHaloPartition(tf.test.TestCase):
    def setUp(self):
        super(TestHaloPartition, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestHaloPartition, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_halo_x10(self):  # same as window partition
        inputs = np.arange(1 * 8 * 12 * 1, dtype='float32').reshape([1, 8, 12, 1])
        height, width = inputs.shape[1:3]
        size, halo_size, dilation_rate = 4, 4, 1

        expected = np.array([
            0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 4, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40,
            41, 42, 43, 8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 48, 49, 50, 51, 60, 61, 62, 63,
            72, 73, 74, 75, 84, 85, 86, 87, 52, 53, 54, 55, 64, 65, 66, 67, 76, 77, 78, 79, 88, 89, 90, 91, 56, 57, 58,
            59, 68, 69, 70, 71, 80, 81, 82, 83, 92, 93, 94, 95], 'int64').reshape([1, 6, 16, 1])

        result = halo_partition(inputs, height, width, size, halo_size, dilation_rate)
        result = self.evaluate(result)

        self.assertAllEqual(expected, result)

    def test_halo_x15(self):
        inputs = np.arange(1 * 8 * 12 * 1, dtype='float32').reshape([1, 8, 12, 1])
        height, width = inputs.shape[1:3]
        size, halo_size, dilation_rate = 4, 6, 1

        expected = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 12, 13, 14, 15, 16, 0, 24, 25, 26, 27, 28, 0, 36, 37, 38, 39, 40, 0,
            48, 49, 50, 51, 52, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31, 32, 39,
            40, 41, 42, 43, 44, 51, 52, 53, 54, 55, 56, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10, 11, 0, 19, 20, 21, 22, 23, 0, 31,
            32, 33, 34, 35, 0, 43, 44, 45, 46, 47, 0, 55, 56, 57, 58, 59, 0, 0, 36, 37, 38, 39, 40, 0, 48, 49, 50, 51,
            52, 0, 60, 61, 62, 63, 64, 0, 72, 73, 74, 75, 76, 0, 84, 85, 86, 87, 88, 0, 0, 0, 0, 0, 0, 39, 40, 41, 42,
            43, 44, 51, 52, 53, 54, 55, 56, 63, 64, 65, 66, 67, 68, 75, 76, 77, 78, 79, 80, 87, 88, 89, 90, 91, 92, 0,
            0, 0, 0, 0, 0, 43, 44, 45, 46, 47, 0, 55, 56, 57, 58, 59, 0, 67, 68, 69, 70, 71, 0, 79, 80, 81, 82, 83, 0,
            91, 92, 93, 94, 95, 0, 0, 0, 0, 0, 0, 0], 'int64').reshape([1, 6, 36, 1])

        result = halo_partition(inputs, height, width, size, halo_size, dilation_rate)
        result = self.evaluate(result)

        self.assertAllEqual(expected, result)

    def test_halo_x20(self):
        inputs = np.arange(1 * 8 * 12 * 1, dtype='float32').reshape([1, 8, 12, 1])
        height, width = inputs.shape[1:3]
        size, halo_size, dilation_rate = 4, 8, 1

        expected = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 12, 13, 14, 15, 16, 17, 0, 0,
            24, 25, 26, 27, 28, 29, 0, 0, 36, 37, 38, 39, 40, 41, 0, 0, 48, 49, 50, 51, 52, 53, 0, 0, 60, 61, 62, 63,
            64, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20,
            21, 26, 27, 28, 29, 30, 31, 32, 33, 38, 39, 40, 41, 42, 43, 44, 45, 50, 51, 52, 53, 54, 55, 56, 57, 62, 63,
            64, 65, 66, 67, 68, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 18, 19,
            20, 21, 22, 23, 0, 0, 30, 31, 32, 33, 34, 35, 0, 0, 42, 43, 44, 45, 46, 47, 0, 0, 54, 55, 56, 57, 58, 59, 0,
            0, 66, 67, 68, 69, 70, 71, 0, 0, 0, 0, 24, 25, 26, 27, 28, 29, 0, 0, 36, 37, 38, 39, 40, 41, 0, 0, 48, 49,
            50, 51, 52, 53, 0, 0, 60, 61, 62, 63, 64, 65, 0, 0, 72, 73, 74, 75, 76, 77, 0, 0, 84, 85, 86, 87, 88, 89, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 27, 28, 29, 30, 31, 32, 33, 38, 39, 40, 41, 42, 43, 44, 45,
            50, 51, 52, 53, 54, 55, 56, 57, 62, 63, 64, 65, 66, 67, 68, 69, 74, 75, 76, 77, 78, 79, 80, 81, 86, 87, 88,
            89, 90, 91, 92, 93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 31, 32, 33, 34, 35, 0, 0, 42, 43,
            44, 45, 46, 47, 0, 0, 54, 55, 56, 57, 58, 59, 0, 0, 66, 67, 68, 69, 70, 71, 0, 0, 78, 79, 80, 81, 82, 83, 0,
            0, 90, 91, 92, 93, 94, 95, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ], 'int64').reshape([1, 6, 64, 1])

        result = halo_partition(inputs, height, width, size, halo_size, dilation_rate)
        result = self.evaluate(result)

        self.assertAllEqual(expected, result)

    def test_halo_x10_d2(self):  # same as window partition with dilation
        inputs = np.arange(1 * 8 * 16 * 1, dtype='float32').reshape([1, 8, 16, 1])
        height, width = inputs.shape[1:3]
        size, halo_size, dilation_rate = 4, 4, 2

        expected = np.array([
            0, 2, 4, 6, 32, 34, 36, 38, 64, 66, 68, 70, 96, 98, 100, 102, 1, 3, 5, 7, 33, 35, 37, 39, 65, 67, 69, 71,
            97, 99, 101, 103, 8, 10, 12, 14, 40, 42, 44, 46, 72, 74, 76, 78, 104, 106, 108, 110, 9, 11, 13, 15, 41, 43,
            45, 47, 73, 75, 77, 79, 105, 107, 109, 111, 16, 18, 20, 22, 48, 50, 52, 54, 80, 82, 84, 86, 112, 114, 116,
            118, 17, 19, 21, 23, 49, 51, 53, 55, 81, 83, 85, 87, 113, 115, 117, 119, 24, 26, 28, 30, 56, 58, 60, 62, 88,
            90, 92, 94, 120, 122, 124, 126, 25, 27, 29, 31, 57, 59, 61, 63, 89, 91, 93, 95, 121, 123, 125, 127],
            'int64').reshape([1, 8, 16, 1])

        result = halo_partition(inputs, height, width, size, halo_size, dilation_rate)
        result = self.evaluate(result)

        self.assertAllEqual(expected, result)

    def test_halo_x20_d2(self):
        inputs = np.arange(1 * 8 * 16 * 1, dtype='float32').reshape([1, 8, 16, 1])
        height, width = inputs.shape[1:3]
        size, halo_size, dilation_rate = 4, 8, 2

        expected = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 6, 8, 10, 0, 0, 32, 34, 36, 38, 40, 42, 0, 0,
            64, 66, 68, 70, 72, 74, 0, 0, 96, 98, 100, 102, 104, 106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 7, 9, 11, 0, 0, 33, 35, 37, 39, 41, 43, 0, 0,
            65, 67, 69, 71, 73, 75, 0, 0, 97, 99, 101, 103, 105, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 8, 10, 12, 14, 0, 0, 36, 38, 40, 42, 44, 46, 0, 0, 68,
            70, 72, 74, 76, 78, 0, 0, 100, 102, 104, 106, 108, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 7, 9, 11, 13, 15, 0, 0, 37, 39, 41, 43, 45, 47, 0, 0,
            69, 71, 73, 75, 77, 79, 0, 0, 101, 103, 105, 107, 109, 111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 18, 20, 22, 24, 26, 0, 0, 48, 50, 52, 54,
            56, 58, 0, 0, 80, 82, 84, 86, 88, 90, 0, 0, 112, 114, 116, 118, 120, 122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 19, 21, 23, 25, 27, 0, 0, 49, 51,
            53, 55, 57, 59, 0, 0, 81, 83, 85, 87, 89, 91, 0, 0, 113, 115, 117, 119, 121, 123, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 22, 24, 26, 28, 30, 0, 0, 52, 54,
            56, 58, 60, 62, 0, 0, 84, 86, 88, 90, 92, 94, 0, 0, 116, 118, 120, 122, 124, 126, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 23, 25, 27, 29, 31, 0, 0, 53,
            55, 57, 59, 61, 63, 0, 0, 85, 87, 89, 91, 93, 95, 0, 0, 117, 119, 121, 123, 125, 127, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'int64').reshape([1, 8, 64, 1])

        result = halo_partition(inputs, height, width, size, halo_size, dilation_rate)
        result = self.evaluate(result)

        self.assertAllEqual(expected, result)

    def test_center(self):
        inputs = np.arange(1 * 24 * 48 * 1, dtype='float32').reshape([1, 24, 48, 1])
        height, width = inputs.shape[1:3]
        size = 4

        for halo_size in [4, 6, 8]:
            pad = (halo_size - size) // 2

            for dilation_rate in [1, 2, 3]:
                expected = partition_apply(inputs, height, width, 'window_size', size, dilation_rate)

                result = halo_partition(inputs, height, width, size, halo_size, dilation_rate)
                result = self.evaluate(result)
                if pad > 0:
                    result = result.reshape(result.shape[:2] + (halo_size, halo_size))
                    result = result[:, :, pad:-pad, pad:-pad]
                    result = result.reshape(result.shape[:2] + (size ** 2, 1))

                self.assertAllEqual(expected, result)


@test_util.run_all_in_graph_and_eager_modes
class TestHaloPartitionFused(tf.test.TestCase):
    def setUp(self):
        super(TestHaloPartitionFused, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestHaloPartitionFused, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_apply(self):
        inputs = np.arange(3 * 24 * 48 * 4 * 2 * 5, dtype='float32').reshape([3, 24, 48, 4 * 2 * 5])
        height, width = inputs.shape[1:3]
        size, channels = 4, 20

        for halo_size in [4, 6, 8]:
            for dilation_rate in [1, 2, 3]:
                for num_heads in [1, 2, 4]:
                    expected = halo_partition(inputs, height, width, size, halo_size, dilation_rate)
                    expected = tf.reshape(expected, expected.shape[:-1] + (num_heads, 2 * channels // num_heads))
                    expected = tf.transpose(expected, [0, 1, 3, 2, 4])
                    expected = self.evaluate(expected)

                    result = halo_partition_fused(inputs, height, width, size, halo_size, num_heads, dilation_rate)
                    result = self.evaluate(result)

                    self.assertAllEqual(expected, result)

    def test_center(self):
        inputs = np.arange(3 * 24 * 48 * 4 * 2 * 5, dtype='float32').reshape([3, 24, 48, 4 * 2 * 5])
        height, width = inputs.shape[1:3]
        size, channels = 4, 20

        for halo_size in [4, 6, 8]:
            pad = (halo_size - size) // 2
            for dilation_rate in [1, 2, 3]:
                for num_heads in [1, 2, 4]:
                    expected = partition_apply_fused(
                        inputs, height, width, 'window_size', size, num_heads, dilation_rate)
                    expected = self.evaluate(expected)

                    result = halo_partition_fused(inputs, height, width, size, halo_size, num_heads, dilation_rate)
                    result = self.evaluate(result)
                    if pad > 0:
                        result = result.reshape(result.shape[:3] + (halo_size, halo_size, -1))
                        result = result[:, :, :, pad:-pad, pad:-pad]
                        result = result.reshape(result.shape[:3] + (size ** 2, -1))

                    self.assertAllEqual(expected, result)


if __name__ == '__main__':
    tf.test.main()

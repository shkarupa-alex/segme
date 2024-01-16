import numpy as np
import tensorflow as tf
from keras import mixed_precision
from keras.src.testing_infra import test_combinations, test_utils
from segme.common.attn.swin import SwinAttention


@test_combinations.run_all_keras_modes
class TestSwinAttention(test_combinations.TestCase):
    def setUp(self):
        super(TestSwinAttention, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSwinAttention, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SwinAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 0, 'qk_units': None,
                'qkv_bias': True, 'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SwinAttention,
            kwargs={
                'current_window': 8, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 0, 'qk_units': None,
                'qkv_bias': True, 'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 14, 18, 4],
            input_dtype='float32',
            expected_output_shape=[None, 14, 18, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SwinAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 4, 'shift_mode': 0, 'qk_units': None,
                'qkv_bias': True, 'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SwinAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 1, 'qk_units': None,
                'qkv_bias': True, 'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SwinAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 2, 'qk_units': None,
                'qkv_bias': True, 'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SwinAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 3, 'qk_units': None,
                'qkv_bias': True, 'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SwinAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 4, 'qk_units': None,
                'qkv_bias': True, 'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SwinAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 0, 'qk_units': 4,
                'qkv_bias': True, 'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SwinAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 0, 'qk_units': None,
                'qkv_bias': False, 'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SwinAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 0, 'qk_units': None,
                'qkv_bias': True, 'cpb_units': 384, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            SwinAttention,
            kwargs={
                'current_window': 6, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 0, 'qk_units': None,
                'qkv_bias': True, 'cpb_units': 512, 'proj_bias': False},
            input_shape=[2, 16, 16, 4],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float16'
        )

    def test_shift_pad(self):
        test_utils.layer_test(
            SwinAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 1, 'qk_units': None,
                'qkv_bias': True, 'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 14, 15, 4],
            input_dtype='float32',
            expected_output_shape=[None, 14, 15, 4],
            expected_output_dtype='float32'
        )

    def test_small(self):
        test_utils.layer_test(
            SwinAttention,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 1, 'shift_mode': 3, 'qk_units': None,
                'qkv_bias': True, 'cpb_units': 512, 'proj_bias': True},
            input_shape=[2, 1, 2, 4],
            input_dtype='float32',
            expected_output_shape=[None, 1, 2, 4],
            expected_output_dtype='float32'
        )

    def test_mask_shift_0_no_pad(self):
        inputs = np.ones([2, 8, 12, 3])
        layer = SwinAttention(4, 4, 1, 0)
        layer.build(inputs.shape)
        layer.rel_bias = lambda x: 0.

        mask = layer.attn_mask(inputs.shape[1:-1], (0, 0, 0, 0), False, [0, 0])
        mask = self.evaluate(mask)

        self.assertTrue((mask == 0.).all())

    def test_mask_shift_0_pad(self):
        inputs = np.ones([2, 7, 9, 3])
        layer = SwinAttention(4, 4, 1, 0)
        layer.build(inputs.shape)
        layer.rel_bias = lambda x: 0.

        mask = layer.attn_mask([8, 12], (0, 1, 1, 2), False, [0, 0])
        mask = self.evaluate(mask)
        mask = mask.reshape(2, 3, 4, 4).transpose(0, 2, 1, 3).reshape(8, 12)
        mask = (mask == 0.).astype('int32')

        self.assertTrue((mask[-1] == 0).all())
        self.assertTrue((mask[:, 0] == 0).all())
        self.assertTrue((mask[:, -2:] == 0).all())
        self.assertTrue((mask[:-1, 1:-2] == 1).all())

    def test_mask_shift_1_no_pad(self):
        inputs = np.ones([2, 8, 12, 3])
        layer = SwinAttention(4, 4, 1, 1)
        layer.build(inputs.shape)
        layer.rel_bias = lambda x: 0.

        mask = layer.attn_mask(inputs.shape[1:-1], (0, 0, 0, 0), True, [2, 2])
        mask = self.evaluate(mask)
        mask = (mask == 0.).astype('int32').reshape(6, 16, 16)

        self.assertTrue((mask[:2] == 1).all())
        self.assertAllEqual(mask[2], np.array([
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]], 'int32'))
        self.assertTrue((mask[3:4, :8, :8] == 1).all())
        self.assertTrue((mask[3:4, :8, 8:] == 0).all())
        self.assertTrue((mask[3:4, 8:, :8] == 0).all())
        self.assertTrue((mask[3:4, 8:, 8:] == 1).all())
        self.assertAllEqual(mask[5], np.array([
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]], 'int32'))

    def test_mask_shift_1_pad(self):
        inputs = np.ones([2, 7, 9, 3])
        layer = SwinAttention(4, 4, 1, 1)
        layer.build(inputs.shape)
        layer.rel_bias = lambda x: 0.

        mask = layer.attn_mask([8, 12], (0, 1, 1, 2), True, [2, 2])
        mask = self.evaluate(mask)
        mask = (mask == 0.).astype('int32').reshape(6, 4, 4, 4, 4)

        # top left window
        self.assertTrue((mask[0, :, :1] == np.array([[[
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]]]], 'int32')).all())
        self.assertTrue((mask[0, :, 1:] == np.array([[[
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1]]]]).all(), 'int32'))

        # bottom right window
        self.assertTrue((mask[5, 0, :2] == np.array([[[
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]]], 'int32')).all())
        self.assertTrue((mask[5, 0, 2:] == np.array([[[
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1]]]], 'int32')).all())
        self.assertTrue((mask[5, 1:3, :2] == np.array([[[
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0]]]], 'int32')).all())
        self.assertTrue((mask[5, 1:3, 2:] == np.array([[[
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1]]]], 'int32')).all())
        self.assertTrue((mask[5, 3] == np.array([[[
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1]]]], 'int32')).all())

    def test_mask_shift_2_no_pad(self):
        inputs = np.ones([2, 8, 12, 3])
        layer = SwinAttention(4, 4, 1, 2)
        layer.build(inputs.shape)
        layer.rel_bias = lambda x: 0.

        mask = layer.attn_mask(inputs.shape[1:-1], (0, 0, 0, 0), True, [2, 2])
        mask = self.evaluate(mask)
        mask = (mask == 0.).astype('int32').reshape(6, 16, 16)

        self.assertAllEqual(mask[0], np.array([
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]], 'int32'))
        self.assertTrue((mask[1:3] == 1).all())
        self.assertAllEqual(mask[3], np.array([
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]], 'int32'))
        self.assertTrue((mask[4:, :8, :8] == 1).all())
        self.assertTrue((mask[4:, :8, 8:] == 0).all())
        self.assertTrue((mask[4:, 8:, :8] == 0).all())
        self.assertTrue((mask[4:, 8:, 8:] == 1).all())

    def test_mask_shift_2_pad(self):
        inputs = np.ones([2, 7, 9, 3])
        layer = SwinAttention(4, 4, 1, 2)
        layer.build(inputs.shape)
        layer.rel_bias = lambda x: 0.

        mask = layer.attn_mask([8, 12], (0, 1, 1, 2), True, [2, 2])
        mask = self.evaluate(mask)
        mask = (mask == 0.).astype('int32').reshape(6, 4, 4, 4, 4)

        # top right window
        self.assertTrue((mask[2, :, :2] == np.array([[[
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0]]]], 'int32')).all())
        self.assertTrue((mask[2, :, 2:] == np.array([[[
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1]]]], 'int32')).all())

        # bottom left window
        self.assertTrue((mask[3, :, 0] == np.array([[
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1]]], 'int32')).all())
        self.assertTrue((mask[3, 0, 1:3] == np.array([[
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]]], 'int32')).all())
        self.assertTrue((mask[3, 0, 3] == np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]], 'int32')).all())
        self.assertTrue((mask[3, 1:3, 1:3] == np.array([[[
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]]]], 'int32')).all())
        self.assertTrue((mask[3, 1:3, 3] == np.array([[
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]]], 'int32')).all())
        self.assertTrue((mask[3, 3, 1:] == np.array([[
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1]]], 'int32')).all())

    def test_mask_shift_3_pad_to_min_size(self):
        inputs = np.ones([2, 3, 5, 3])
        layer = SwinAttention(4, 4, 1, 3)
        layer.build(inputs.shape)
        layer.rel_bias = lambda x: 0.

        mask = layer.attn_mask([4, 8], (0, 1, 1, 2), True, [2, 2])
        mask = self.evaluate(mask)
        mask = (mask == 0.).astype('int32').reshape(2, 4, 4, 4, 4)

        # left window
        self.assertTrue((mask[0, :, 0] == np.array([[
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1]]], 'int32')).all())
        self.assertTrue((mask[0, :3, 1:3] == np.array([[
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]]], 'int32')).all())
        self.assertTrue((mask[0, :3, 3] == np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]], 'int32')).all())
        self.assertTrue((mask[0, 3] == np.array([[
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1]]], 'int32')).all())

        # right window
        self.assertTrue((mask[1, :3, :2] == np.array([[
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0]]], 'int32')).all())
        self.assertTrue((mask[1, :3, 2:] == np.array([[
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1]]], 'int32')).all())
        self.assertTrue((mask[1, 3] == np.array([[
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1]]], 'int32')).all())


if __name__ == '__main__':
    tf.test.main()

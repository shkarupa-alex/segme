import numpy as np
import tensorflow as tf
from keras import mixed_precision
from keras.src.testing_infra import test_combinations, test_utils
from segme.common.attn import DHMSA, SWMSA, GGMSA, RelativeBias, CHMSA


@test_combinations.run_all_keras_modes
class TestDHMSA(test_combinations.TestCase):
    def setUp(self):
        super(TestDHMSA, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestDHMSA, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            DHMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'dilation_rate': 1, 'qkv_bias': True,
                'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DHMSA,
            kwargs={
                'current_window': 8, 'pretrain_window': 4, 'num_heads': 2, 'dilation_rate': 1, 'qkv_bias': True,
                'proj_bias': True},
            input_shape=[2, 14, 18, 4],
            input_dtype='float32',
            expected_output_shape=[None, 14, 18, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DHMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 4, 'dilation_rate': 1, 'qkv_bias': True,
                'proj_bias': True},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DHMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'dilation_rate': 2, 'qkv_bias': True,
                'proj_bias': True},
            input_shape=[2, 13, 19, 4],
            input_dtype='float32',
            expected_output_shape=[None, 13, 19, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            DHMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'dilation_rate': 1, 'qkv_bias': False,
                'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            DHMSA,
            kwargs={'current_window': 8, 'pretrain_window': 4, 'num_heads': 2, 'dilation_rate': 2, 'qkv_bias': True,
                    'proj_bias': False},
            input_shape=[2, 16, 16, 4],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float16'
        )


@test_combinations.run_all_keras_modes
class TestSWMSA(test_combinations.TestCase):
    def setUp(self):
        super(TestSWMSA, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestSWMSA, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SWMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 0, 'use_dw': False,
                'qkv_bias': True, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SWMSA,
            kwargs={
                'current_window': 8, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 0, 'use_dw': False,
                'qkv_bias': True, 'proj_bias': True},
            input_shape=[2, 14, 18, 4],
            input_dtype='float32',
            expected_output_shape=[None, 14, 18, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SWMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 4, 'shift_mode': 0, 'use_dw': False,
                'qkv_bias': True, 'proj_bias': True},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SWMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 1, 'use_dw': False,
                'qkv_bias': True, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SWMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 2, 'use_dw': False,
                'qkv_bias': True, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SWMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 3, 'use_dw': False,
                'qkv_bias': True, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SWMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 4, 'use_dw': False,
                'qkv_bias': True, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SWMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 0, 'use_dw': True,
                'qkv_bias': True, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SWMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 0, 'use_dw': False,
                'qkv_bias': False, 'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            SWMSA,
            kwargs={
                'current_window': 6, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 0, 'use_dw': False,
                'qkv_bias': True, 'proj_bias': False},
            input_shape=[2, 16, 16, 4],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float16'
        )

    def test_shift_pad(self):
        test_utils.layer_test(
            SWMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'shift_mode': 1, 'use_dw': False,
                'qkv_bias': True, 'proj_bias': True},
            input_shape=[2, 14, 15, 4],
            input_dtype='float32',
            expected_output_shape=[None, 14, 15, 4],
            expected_output_dtype='float32'
        )

    def test_small(self):
        test_utils.layer_test(
            SWMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 1, 'shift_mode': 3, 'use_dw': False,
                'qkv_bias': True, 'proj_bias': True},
            input_shape=[2, 1, 2, 4],
            input_dtype='float32',
            expected_output_shape=[None, 1, 2, 4],
            expected_output_dtype='float32'
        )

    def test_mask_shift_0_no_pad(self):
        inputs = np.ones([2, 8, 12, 3])
        layer = SWMSA(4, 4, 1, 0)
        layer.build(inputs.shape)
        layer.rel_bias = lambda x: 0.

        mask = layer.attn_mask(inputs.shape[:-1], (0, 0, 0, 0), False, [0, 0])
        mask = self.evaluate(mask)

        self.assertTrue((mask == 0.).all())

    def test_mask_shift_0_pad(self):
        inputs = np.ones([2, 7, 9, 3])
        layer = SWMSA(4, 4, 1, 0)
        layer.build(inputs.shape)
        layer.rel_bias = lambda x: 0.

        mask = layer.attn_mask([2, 8, 12], (0, 1, 1, 2), False, [0, 0])
        mask = self.evaluate(mask)
        mask = mask.reshape(2, 3, 4, 4).transpose(0, 2, 1, 3).reshape(8, 12)
        mask = (mask == 0.).astype('int32')

        self.assertTrue((mask[-1] == 0).all())
        self.assertTrue((mask[:, 0] == 0).all())
        self.assertTrue((mask[:, -2:] == 0).all())
        self.assertTrue((mask[:-1, 1:-2] == 1).all())

    def test_mask_shift_1_no_pad(self):
        inputs = np.ones([2, 8, 12, 3])
        layer = SWMSA(4, 4, 1, 1)
        layer.build(inputs.shape)
        layer.rel_bias = lambda x: 0.

        mask = layer.attn_mask(inputs.shape[:-1], (0, 0, 0, 0), True, [2, 2])
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
        layer = SWMSA(4, 4, 1, 1)
        layer.build(inputs.shape)
        layer.rel_bias = lambda x: 0.

        mask = layer.attn_mask([2, 8, 12], (0, 1, 1, 2), True, [2, 2])
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
        layer = SWMSA(4, 4, 1, 2)
        layer.build(inputs.shape)
        layer.rel_bias = lambda x: 0.

        mask = layer.attn_mask(inputs.shape[:-1], (0, 0, 0, 0), True, [2, 2])
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
        layer = SWMSA(4, 4, 1, 2)
        layer.build(inputs.shape)
        layer.rel_bias = lambda x: 0.

        mask = layer.attn_mask([2, 8, 12], (0, 1, 1, 2), True, [2, 2])
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
        layer = SWMSA(4, 4, 1, 3)
        layer.build(inputs.shape)
        layer.rel_bias = lambda x: 0.

        mask = layer.attn_mask([2, 4, 8], (0, 1, 1, 2), True, [2, 2])
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


@test_combinations.run_all_keras_modes
class TestGGMSA(test_combinations.TestCase):
    def setUp(self):
        super(TestGGMSA, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestGGMSA, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            GGMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'use_dw': False, 'qkv_bias': True,
                'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GGMSA,
            kwargs={
                'current_window': 8, 'pretrain_window': 4, 'num_heads': 2, 'use_dw': False, 'qkv_bias': True,
                'proj_bias': True},
            input_shape=[2, 14, 18, 4],
            input_dtype='float32',
            expected_output_shape=[None, 14, 18, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GGMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 4, 'use_dw': False, 'qkv_bias': True,
                'proj_bias': True},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GGMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 4, 'use_dw': True, 'qkv_bias': True,
                'proj_bias': True},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GGMSA,
            kwargs={
                'current_window': 4, 'pretrain_window': 4, 'num_heads': 2, 'use_dw': False, 'qkv_bias': False,
                'proj_bias': True},
            input_shape=[2, 15, 17, 4],
            input_dtype='float32',
            expected_output_shape=[None, 15, 17, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            GGMSA,
            kwargs={
                'current_window': 6, 'pretrain_window': 4, 'num_heads': 2, 'use_dw': False, 'qkv_bias': True,
                'proj_bias': False},
            input_shape=[2, 16, 16, 4],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float16'
        )


@test_combinations.run_all_keras_modes
class TestRelativeBias(test_combinations.TestCase):
    def setUp(self):
        super(TestRelativeBias, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestRelativeBias, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            RelativeBias,
            kwargs={'query_window': 8, 'pretrain_window': 8, 'key_window': 8, 'num_heads': 2},
            input_data=np.zeros([1]),
            expected_output_shape=[1, 1, 2, 64, 64],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            RelativeBias,
            kwargs={'query_window': 7, 'pretrain_window': 7, 'key_window': 7, 'num_heads': 2},
            input_data=np.zeros([1]),
            expected_output_shape=[1, 1, 2, 49, 49],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            RelativeBias,
            kwargs={'query_window': 12, 'pretrain_window': 8, 'key_window': 12, 'num_heads': 2},
            input_data=np.zeros([1]),
            expected_output_shape=[1, 1, 2, 144, 144],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            RelativeBias,
            kwargs={'query_window': 8, 'pretrain_window': 8, 'key_window': 16, 'num_heads': 2},
            input_data=np.zeros([1]),
            expected_output_shape=[1, 1, 2, 64, 256],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            RelativeBias,
            kwargs={'query_window': 8, 'pretrain_window': 8, 'key_window': 8, 'num_heads': 2},
            input_data=np.zeros([1]),
            expected_output_shape=[1, 1, 2, 64, 64],
            expected_output_dtype='float16'
        )

    def test_value(self):
        expected_table = np.array([
            -1.057, -1.057, -1.057, -0.991, -1.057, -0.916, -1.057, -0.826, -1.057, -0.716, -1.057, -0.572, -1.057,
            -0.367, -1.057, 0.0, -1.057, 0.367, -1.057, 0.572, -1.057, 0.716, -1.057, 0.826, -1.057, 0.916, -1.057,
            0.991, -1.057, 1.057, -0.991, -1.057, -0.991, -0.991, -0.991, -0.916, -0.991, -0.826, -0.991, -0.716,
            -0.991, -0.572, -0.991, -0.367, -0.991, 0.0, -0.991, 0.367, -0.991, 0.572, -0.991, 0.716, -0.991, 0.826,
            -0.991, 0.916, -0.991, 0.991, -0.991, 1.057, -0.916, -1.057, -0.916, -0.991, -0.916, -0.916, -0.916, -0.826,
            -0.916, -0.716, -0.916, -0.572, -0.916, -0.367, -0.916, 0.0, -0.916, 0.367, -0.916, 0.572, -0.916, 0.716,
            -0.916, 0.826, -0.916, 0.916, -0.916, 0.991, -0.916, 1.057, -0.826, -1.057, -0.826, -0.991, -0.826, -0.916,
            -0.826, -0.826, -0.826, -0.716, -0.826, -0.572, -0.826, -0.367, -0.826, 0.0, -0.826, 0.367, -0.826, 0.572,
            -0.826, 0.716, -0.826, 0.826, -0.826, 0.916, -0.826, 0.991, -0.826, 1.057, -0.716, -1.057, -0.716, -0.991,
            -0.716, -0.916, -0.716, -0.826, -0.716, -0.716, -0.716, -0.572, -0.716, -0.367, -0.716, 0.0, -0.716, 0.367,
            -0.716, 0.572, -0.716, 0.716, -0.716, 0.826, -0.716, 0.916, -0.716, 0.991, -0.716, 1.057, -0.572, -1.057,
            -0.572, -0.991, -0.572, -0.916, -0.572, -0.826, -0.572, -0.716, -0.572, -0.572, -0.572, -0.367, -0.572, 0.0,
            -0.572, 0.367, -0.572, 0.572, -0.572, 0.716, -0.572, 0.826, -0.572, 0.916, -0.572, 0.991, -0.572, 1.057,
            -0.367, -1.057, -0.367, -0.991, -0.367, -0.916, -0.367, -0.826, -0.367, -0.716, -0.367, -0.572, -0.367,
            -0.367, -0.367, 0.0, -0.367, 0.367, -0.367, 0.572, -0.367, 0.716, -0.367, 0.826, -0.367, 0.916, -0.367,
            0.991, -0.367, 1.057, 0.0, -1.057, 0.0, -0.991, 0.0, -0.916, 0.0, -0.826, 0.0, -0.716, 0.0, -0.572, 0.0,
            -0.367, 0.0, 0.0, 0.0, 0.367, 0.0, 0.572, 0.0, 0.716, 0.0, 0.826, 0.0, 0.916, 0.0, 0.991, 0.0, 1.057, 0.367,
            -1.057, 0.367, -0.991, 0.367, -0.916, 0.367, -0.826, 0.367, -0.716, 0.367, -0.572, 0.367, -0.367, 0.367,
            0.0, 0.367, 0.367, 0.367, 0.572, 0.367, 0.716, 0.367, 0.826, 0.367, 0.916, 0.367, 0.991, 0.367, 1.057,
            0.572, -1.057, 0.572, -0.991, 0.572, -0.916, 0.572, -0.826, 0.572, -0.716, 0.572, -0.572, 0.572, -0.367,
            0.572, 0.0, 0.572, 0.367, 0.572, 0.572, 0.572, 0.716, 0.572, 0.826, 0.572, 0.916, 0.572, 0.991, 0.572,
            1.057, 0.716, -1.057, 0.716, -0.991, 0.716, -0.916, 0.716, -0.826, 0.716, -0.716, 0.716, -0.572, 0.716,
            -0.367, 0.716, 0.0, 0.716, 0.367, 0.716, 0.572, 0.716, 0.716, 0.716, 0.826, 0.716, 0.916, 0.716, 0.991,
            0.716, 1.057, 0.826, -1.057, 0.826, -0.991, 0.826, -0.916, 0.826, -0.826, 0.826, -0.716, 0.826, -0.572,
            0.826, -0.367, 0.826, 0.0, 0.826, 0.367, 0.826, 0.572, 0.826, 0.716, 0.826, 0.826, 0.826, 0.916, 0.826,
            0.991, 0.826, 1.057, 0.916, -1.057, 0.916, -0.991, 0.916, -0.916, 0.916, -0.826, 0.916, -0.716, 0.916,
            -0.572, 0.916, -0.367, 0.916, 0.0, 0.916, 0.367, 0.916, 0.572, 0.916, 0.716, 0.916, 0.826, 0.916, 0.916,
            0.916, 0.991, 0.916, 1.057, 0.991, -1.057, 0.991, -0.991, 0.991, -0.916, 0.991, -0.826, 0.991, -0.716,
            0.991, -0.572, 0.991, -0.367, 0.991, 0.0, 0.991, 0.367, 0.991, 0.572, 0.991, 0.716, 0.991, 0.826, 0.991,
            0.916, 0.991, 0.991, 0.991, 1.057, 1.057, -1.057, 1.057, -0.991, 1.057, -0.916, 1.057, -0.826, 1.057,
            -0.716, 1.057, -0.572, 1.057, -0.367, 1.057, 0.0, 1.057, 0.367, 1.057, 0.572, 1.057, 0.716, 1.057, 0.826,
            1.057, 0.916, 1.057, 0.991, 1.057, 1.057], 'float32').reshape([-1, 2])
        expected_index = np.array([
            112, 111, 110, 109, 108, 107, 106, 105, 97, 96, 95, 94, 93, 92, 91, 90, 82, 81, 80, 79, 78, 77, 76, 75, 67,
            66, 65, 64, 63, 62, 61, 60, 52, 51, 50, 49, 48, 47, 46, 45, 37, 36, 35, 34, 33, 32, 31, 30, 22, 21, 20, 19,
            18, 17, 16, 15, 7, 6, 5, 4, 3, 2, 1, 0, 113, 112, 111, 110, 109, 108, 107, 106, 98, 97, 96, 95, 94, 93, 92,
            91, 83, 82, 81, 80, 79, 78, 77, 76, 68, 67, 66, 65, 64, 63, 62, 61, 53, 52, 51, 50, 49, 48, 47, 46, 38, 37,
            36, 35, 34, 33, 32, 31, 23, 22, 21, 20, 19, 18, 17, 16, 8, 7, 6, 5, 4, 3, 2, 1, 114, 113, 112, 111, 110,
            109, 108, 107, 99, 98, 97, 96, 95, 94, 93, 92, 84, 83, 82, 81, 80, 79, 78, 77, 69, 68, 67, 66, 65, 64, 63,
            62, 54, 53, 52, 51, 50, 49, 48, 47, 39, 38, 37, 36, 35, 34, 33, 32, 24, 23, 22, 21, 20, 19, 18, 17, 9, 8, 7,
            6, 5, 4, 3, 2, 115, 114, 113, 112, 111, 110, 109, 108, 100, 99, 98, 97, 96, 95, 94, 93, 85, 84, 83, 82, 81,
            80, 79, 78, 70, 69, 68, 67, 66, 65, 64, 63, 55, 54, 53, 52, 51, 50, 49, 48, 40, 39, 38, 37, 36, 35, 34, 33,
            25, 24, 23, 22, 21, 20, 19, 18, 10, 9, 8, 7, 6, 5, 4, 3, 116, 115, 114, 113, 112, 111, 110, 109, 101, 100,
            99, 98, 97, 96, 95, 94, 86, 85, 84, 83, 82, 81, 80, 79, 71, 70, 69, 68, 67, 66, 65, 64, 56, 55, 54, 53, 52,
            51, 50, 49, 41, 40, 39, 38, 37, 36, 35, 34, 26, 25, 24, 23, 22, 21, 20, 19, 11, 10, 9, 8, 7, 6, 5, 4, 117,
            116, 115, 114, 113, 112, 111, 110, 102, 101, 100, 99, 98, 97, 96, 95, 87, 86, 85, 84, 83, 82, 81, 80, 72,
            71, 70, 69, 68, 67, 66, 65, 57, 56, 55, 54, 53, 52, 51, 50, 42, 41, 40, 39, 38, 37, 36, 35, 27, 26, 25, 24,
            23, 22, 21, 20, 12, 11, 10, 9, 8, 7, 6, 5, 118, 117, 116, 115, 114, 113, 112, 111, 103, 102, 101, 100, 99,
            98, 97, 96, 88, 87, 86, 85, 84, 83, 82, 81, 73, 72, 71, 70, 69, 68, 67, 66, 58, 57, 56, 55, 54, 53, 52, 51,
            43, 42, 41, 40, 39, 38, 37, 36, 28, 27, 26, 25, 24, 23, 22, 21, 13, 12, 11, 10, 9, 8, 7, 6, 119, 118, 117,
            116, 115, 114, 113, 112, 104, 103, 102, 101, 100, 99, 98, 97, 89, 88, 87, 86, 85, 84, 83, 82, 74, 73, 72,
            71, 70, 69, 68, 67, 59, 58, 57, 56, 55, 54, 53, 52, 44, 43, 42, 41, 40, 39, 38, 37, 29, 28, 27, 26, 25, 24,
            23, 22, 14, 13, 12, 11, 10, 9, 8, 7, 127, 126, 125, 124, 123, 122, 121, 120, 112, 111, 110, 109, 108, 107,
            106, 105, 97, 96, 95, 94, 93, 92, 91, 90, 82, 81, 80, 79, 78, 77, 76, 75, 67, 66, 65, 64, 63, 62, 61, 60,
            52, 51, 50, 49, 48, 47, 46, 45, 37, 36, 35, 34, 33, 32, 31, 30, 22, 21, 20, 19, 18, 17, 16, 15, 128, 127,
            126, 125, 124, 123, 122, 121, 113, 112, 111, 110, 109, 108, 107, 106, 98, 97, 96, 95, 94, 93, 92, 91, 83,
            82, 81, 80, 79, 78, 77, 76, 68, 67, 66, 65, 64, 63, 62, 61, 53, 52, 51, 50, 49, 48, 47, 46, 38, 37, 36, 35,
            34, 33, 32, 31, 23, 22, 21, 20, 19, 18, 17, 16, 129, 128, 127, 126, 125, 124, 123, 122, 114, 113, 112, 111,
            110, 109, 108, 107, 99, 98, 97, 96, 95, 94, 93, 92, 84, 83, 82, 81, 80, 79, 78, 77, 69, 68, 67, 66, 65, 64,
            63, 62, 54, 53, 52, 51, 50, 49, 48, 47, 39, 38, 37, 36, 35, 34, 33, 32, 24, 23, 22, 21, 20, 19, 18, 17, 130,
            129, 128, 127, 126, 125, 124, 123, 115, 114, 113, 112, 111, 110, 109, 108, 100, 99, 98, 97, 96, 95, 94, 93,
            85, 84, 83, 82, 81, 80, 79, 78, 70, 69, 68, 67, 66, 65, 64, 63, 55, 54, 53, 52, 51, 50, 49, 48, 40, 39, 38,
            37, 36, 35, 34, 33, 25, 24, 23, 22, 21, 20, 19, 18, 131, 130, 129, 128, 127, 126, 125, 124, 116, 115, 114,
            113, 112, 111, 110, 109, 101, 100, 99, 98, 97, 96, 95, 94, 86, 85, 84, 83, 82, 81, 80, 79, 71, 70, 69, 68,
            67, 66, 65, 64, 56, 55, 54, 53, 52, 51, 50, 49, 41, 40, 39, 38, 37, 36, 35, 34, 26, 25, 24, 23, 22, 21, 20,
            19, 132, 131, 130, 129, 128, 127, 126, 125, 117, 116, 115, 114, 113, 112, 111, 110, 102, 101, 100, 99, 98,
            97, 96, 95, 87, 86, 85, 84, 83, 82, 81, 80, 72, 71, 70, 69, 68, 67, 66, 65, 57, 56, 55, 54, 53, 52, 51, 50,
            42, 41, 40, 39, 38, 37, 36, 35, 27, 26, 25, 24, 23, 22, 21, 20, 133, 132, 131, 130, 129, 128, 127, 126, 118,
            117, 116, 115, 114, 113, 112, 111, 103, 102, 101, 100, 99, 98, 97, 96, 88, 87, 86, 85, 84, 83, 82, 81, 73,
            72, 71, 70, 69, 68, 67, 66, 58, 57, 56, 55, 54, 53, 52, 51, 43, 42, 41, 40, 39, 38, 37, 36, 28, 27, 26, 25,
            24, 23, 22, 21, 134, 133, 132, 131, 130, 129, 128, 127, 119, 118, 117, 116, 115, 114, 113, 112, 104, 103,
            102, 101, 100, 99, 98, 97, 89, 88, 87, 86, 85, 84, 83, 82, 74, 73, 72, 71, 70, 69, 68, 67, 59, 58, 57, 56,
            55, 54, 53, 52, 44, 43, 42, 41, 40, 39, 38, 37, 29, 28, 27, 26, 25, 24, 23, 22, 142, 141, 140, 139, 138,
            137, 136, 135, 127, 126, 125, 124, 123, 122, 121, 120, 112, 111, 110, 109, 108, 107, 106, 105, 97, 96, 95,
            94, 93, 92, 91, 90, 82, 81, 80, 79, 78, 77, 76, 75, 67, 66, 65, 64, 63, 62, 61, 60, 52, 51, 50, 49, 48, 47,
            46, 45, 37, 36, 35, 34, 33, 32, 31, 30, 143, 142, 141, 140, 139, 138, 137, 136, 128, 127, 126, 125, 124,
            123, 122, 121, 113, 112, 111, 110, 109, 108, 107, 106, 98, 97, 96, 95, 94, 93, 92, 91, 83, 82, 81, 80, 79,
            78, 77, 76, 68, 67, 66, 65, 64, 63, 62, 61, 53, 52, 51, 50, 49, 48, 47, 46, 38, 37, 36, 35, 34, 33, 32, 31,
            144, 143, 142, 141, 140, 139, 138, 137, 129, 128, 127, 126, 125, 124, 123, 122, 114, 113, 112, 111, 110,
            109, 108, 107, 99, 98, 97, 96, 95, 94, 93, 92, 84, 83, 82, 81, 80, 79, 78, 77, 69, 68, 67, 66, 65, 64, 63,
            62, 54, 53, 52, 51, 50, 49, 48, 47, 39, 38, 37, 36, 35, 34, 33, 32, 145, 144, 143, 142, 141, 140, 139, 138,
            130, 129, 128, 127, 126, 125, 124, 123, 115, 114, 113, 112, 111, 110, 109, 108, 100, 99, 98, 97, 96, 95, 94,
            93, 85, 84, 83, 82, 81, 80, 79, 78, 70, 69, 68, 67, 66, 65, 64, 63, 55, 54, 53, 52, 51, 50, 49, 48, 40, 39,
            38, 37, 36, 35, 34, 33, 146, 145, 144, 143, 142, 141, 140, 139, 131, 130, 129, 128, 127, 126, 125, 124, 116,
            115, 114, 113, 112, 111, 110, 109, 101, 100, 99, 98, 97, 96, 95, 94, 86, 85, 84, 83, 82, 81, 80, 79, 71, 70,
            69, 68, 67, 66, 65, 64, 56, 55, 54, 53, 52, 51, 50, 49, 41, 40, 39, 38, 37, 36, 35, 34, 147, 146, 145, 144,
            143, 142, 141, 140, 132, 131, 130, 129, 128, 127, 126, 125, 117, 116, 115, 114, 113, 112, 111, 110, 102,
            101, 100, 99, 98, 97, 96, 95, 87, 86, 85, 84, 83, 82, 81, 80, 72, 71, 70, 69, 68, 67, 66, 65, 57, 56, 55,
            54, 53, 52, 51, 50, 42, 41, 40, 39, 38, 37, 36, 35, 148, 147, 146, 145, 144, 143, 142, 141, 133, 132, 131,
            130, 129, 128, 127, 126, 118, 117, 116, 115, 114, 113, 112, 111, 103, 102, 101, 100, 99, 98, 97, 96, 88, 87,
            86, 85, 84, 83, 82, 81, 73, 72, 71, 70, 69, 68, 67, 66, 58, 57, 56, 55, 54, 53, 52, 51, 43, 42, 41, 40, 39,
            38, 37, 36, 149, 148, 147, 146, 145, 144, 143, 142, 134, 133, 132, 131, 130, 129, 128, 127, 119, 118, 117,
            116, 115, 114, 113, 112, 104, 103, 102, 101, 100, 99, 98, 97, 89, 88, 87, 86, 85, 84, 83, 82, 74, 73, 72,
            71, 70, 69, 68, 67, 59, 58, 57, 56, 55, 54, 53, 52, 44, 43, 42, 41, 40, 39, 38, 37, 157, 156, 155, 154, 153,
            152, 151, 150, 142, 141, 140, 139, 138, 137, 136, 135, 127, 126, 125, 124, 123, 122, 121, 120, 112, 111,
            110, 109, 108, 107, 106, 105, 97, 96, 95, 94, 93, 92, 91, 90, 82, 81, 80, 79, 78, 77, 76, 75, 67, 66, 65,
            64, 63, 62, 61, 60, 52, 51, 50, 49, 48, 47, 46, 45, 158, 157, 156, 155, 154, 153, 152, 151, 143, 142, 141,
            140, 139, 138, 137, 136, 128, 127, 126, 125, 124, 123, 122, 121, 113, 112, 111, 110, 109, 108, 107, 106, 98,
            97, 96, 95, 94, 93, 92, 91, 83, 82, 81, 80, 79, 78, 77, 76, 68, 67, 66, 65, 64, 63, 62, 61, 53, 52, 51, 50,
            49, 48, 47, 46, 159, 158, 157, 156, 155, 154, 153, 152, 144, 143, 142, 141, 140, 139, 138, 137, 129, 128,
            127, 126, 125, 124, 123, 122, 114, 113, 112, 111, 110, 109, 108, 107, 99, 98, 97, 96, 95, 94, 93, 92, 84,
            83, 82, 81, 80, 79, 78, 77, 69, 68, 67, 66, 65, 64, 63, 62, 54, 53, 52, 51, 50, 49, 48, 47, 160, 159, 158,
            157, 156, 155, 154, 153, 145, 144, 143, 142, 141, 140, 139, 138, 130, 129, 128, 127, 126, 125, 124, 123,
            115, 114, 113, 112, 111, 110, 109, 108, 100, 99, 98, 97, 96, 95, 94, 93, 85, 84, 83, 82, 81, 80, 79, 78, 70,
            69, 68, 67, 66, 65, 64, 63, 55, 54, 53, 52, 51, 50, 49, 48, 161, 160, 159, 158, 157, 156, 155, 154, 146,
            145, 144, 143, 142, 141, 140, 139, 131, 130, 129, 128, 127, 126, 125, 124, 116, 115, 114, 113, 112, 111,
            110, 109, 101, 100, 99, 98, 97, 96, 95, 94, 86, 85, 84, 83, 82, 81, 80, 79, 71, 70, 69, 68, 67, 66, 65, 64,
            56, 55, 54, 53, 52, 51, 50, 49, 162, 161, 160, 159, 158, 157, 156, 155, 147, 146, 145, 144, 143, 142, 141,
            140, 132, 131, 130, 129, 128, 127, 126, 125, 117, 116, 115, 114, 113, 112, 111, 110, 102, 101, 100, 99, 98,
            97, 96, 95, 87, 86, 85, 84, 83, 82, 81, 80, 72, 71, 70, 69, 68, 67, 66, 65, 57, 56, 55, 54, 53, 52, 51, 50,
            163, 162, 161, 160, 159, 158, 157, 156, 148, 147, 146, 145, 144, 143, 142, 141, 133, 132, 131, 130, 129,
            128, 127, 126, 118, 117, 116, 115, 114, 113, 112, 111, 103, 102, 101, 100, 99, 98, 97, 96, 88, 87, 86, 85,
            84, 83, 82, 81, 73, 72, 71, 70, 69, 68, 67, 66, 58, 57, 56, 55, 54, 53, 52, 51, 164, 163, 162, 161, 160,
            159, 158, 157, 149, 148, 147, 146, 145, 144, 143, 142, 134, 133, 132, 131, 130, 129, 128, 127, 119, 118,
            117, 116, 115, 114, 113, 112, 104, 103, 102, 101, 100, 99, 98, 97, 89, 88, 87, 86, 85, 84, 83, 82, 74, 73,
            72, 71, 70, 69, 68, 67, 59, 58, 57, 56, 55, 54, 53, 52, 172, 171, 170, 169, 168, 167, 166, 165, 157, 156,
            155, 154, 153, 152, 151, 150, 142, 141, 140, 139, 138, 137, 136, 135, 127, 126, 125, 124, 123, 122, 121,
            120, 112, 111, 110, 109, 108, 107, 106, 105, 97, 96, 95, 94, 93, 92, 91, 90, 82, 81, 80, 79, 78, 77, 76, 75,
            67, 66, 65, 64, 63, 62, 61, 60, 173, 172, 171, 170, 169, 168, 167, 166, 158, 157, 156, 155, 154, 153, 152,
            151, 143, 142, 141, 140, 139, 138, 137, 136, 128, 127, 126, 125, 124, 123, 122, 121, 113, 112, 111, 110,
            109, 108, 107, 106, 98, 97, 96, 95, 94, 93, 92, 91, 83, 82, 81, 80, 79, 78, 77, 76, 68, 67, 66, 65, 64, 63,
            62, 61, 174, 173, 172, 171, 170, 169, 168, 167, 159, 158, 157, 156, 155, 154, 153, 152, 144, 143, 142, 141,
            140, 139, 138, 137, 129, 128, 127, 126, 125, 124, 123, 122, 114, 113, 112, 111, 110, 109, 108, 107, 99, 98,
            97, 96, 95, 94, 93, 92, 84, 83, 82, 81, 80, 79, 78, 77, 69, 68, 67, 66, 65, 64, 63, 62, 175, 174, 173, 172,
            171, 170, 169, 168, 160, 159, 158, 157, 156, 155, 154, 153, 145, 144, 143, 142, 141, 140, 139, 138, 130,
            129, 128, 127, 126, 125, 124, 123, 115, 114, 113, 112, 111, 110, 109, 108, 100, 99, 98, 97, 96, 95, 94, 93,
            85, 84, 83, 82, 81, 80, 79, 78, 70, 69, 68, 67, 66, 65, 64, 63, 176, 175, 174, 173, 172, 171, 170, 169, 161,
            160, 159, 158, 157, 156, 155, 154, 146, 145, 144, 143, 142, 141, 140, 139, 131, 130, 129, 128, 127, 126,
            125, 124, 116, 115, 114, 113, 112, 111, 110, 109, 101, 100, 99, 98, 97, 96, 95, 94, 86, 85, 84, 83, 82, 81,
            80, 79, 71, 70, 69, 68, 67, 66, 65, 64, 177, 176, 175, 174, 173, 172, 171, 170, 162, 161, 160, 159, 158,
            157, 156, 155, 147, 146, 145, 144, 143, 142, 141, 140, 132, 131, 130, 129, 128, 127, 126, 125, 117, 116,
            115, 114, 113, 112, 111, 110, 102, 101, 100, 99, 98, 97, 96, 95, 87, 86, 85, 84, 83, 82, 81, 80, 72, 71, 70,
            69, 68, 67, 66, 65, 178, 177, 176, 175, 174, 173, 172, 171, 163, 162, 161, 160, 159, 158, 157, 156, 148,
            147, 146, 145, 144, 143, 142, 141, 133, 132, 131, 130, 129, 128, 127, 126, 118, 117, 116, 115, 114, 113,
            112, 111, 103, 102, 101, 100, 99, 98, 97, 96, 88, 87, 86, 85, 84, 83, 82, 81, 73, 72, 71, 70, 69, 68, 67,
            66, 179, 178, 177, 176, 175, 174, 173, 172, 164, 163, 162, 161, 160, 159, 158, 157, 149, 148, 147, 146, 145,
            144, 143, 142, 134, 133, 132, 131, 130, 129, 128, 127, 119, 118, 117, 116, 115, 114, 113, 112, 104, 103,
            102, 101, 100, 99, 98, 97, 89, 88, 87, 86, 85, 84, 83, 82, 74, 73, 72, 71, 70, 69, 68, 67, 187, 186, 185,
            184, 183, 182, 181, 180, 172, 171, 170, 169, 168, 167, 166, 165, 157, 156, 155, 154, 153, 152, 151, 150,
            142, 141, 140, 139, 138, 137, 136, 135, 127, 126, 125, 124, 123, 122, 121, 120, 112, 111, 110, 109, 108,
            107, 106, 105, 97, 96, 95, 94, 93, 92, 91, 90, 82, 81, 80, 79, 78, 77, 76, 75, 188, 187, 186, 185, 184, 183,
            182, 181, 173, 172, 171, 170, 169, 168, 167, 166, 158, 157, 156, 155, 154, 153, 152, 151, 143, 142, 141,
            140, 139, 138, 137, 136, 128, 127, 126, 125, 124, 123, 122, 121, 113, 112, 111, 110, 109, 108, 107, 106, 98,
            97, 96, 95, 94, 93, 92, 91, 83, 82, 81, 80, 79, 78, 77, 76, 189, 188, 187, 186, 185, 184, 183, 182, 174,
            173, 172, 171, 170, 169, 168, 167, 159, 158, 157, 156, 155, 154, 153, 152, 144, 143, 142, 141, 140, 139,
            138, 137, 129, 128, 127, 126, 125, 124, 123, 122, 114, 113, 112, 111, 110, 109, 108, 107, 99, 98, 97, 96,
            95, 94, 93, 92, 84, 83, 82, 81, 80, 79, 78, 77, 190, 189, 188, 187, 186, 185, 184, 183, 175, 174, 173, 172,
            171, 170, 169, 168, 160, 159, 158, 157, 156, 155, 154, 153, 145, 144, 143, 142, 141, 140, 139, 138, 130,
            129, 128, 127, 126, 125, 124, 123, 115, 114, 113, 112, 111, 110, 109, 108, 100, 99, 98, 97, 96, 95, 94, 93,
            85, 84, 83, 82, 81, 80, 79, 78, 191, 190, 189, 188, 187, 186, 185, 184, 176, 175, 174, 173, 172, 171, 170,
            169, 161, 160, 159, 158, 157, 156, 155, 154, 146, 145, 144, 143, 142, 141, 140, 139, 131, 130, 129, 128,
            127, 126, 125, 124, 116, 115, 114, 113, 112, 111, 110, 109, 101, 100, 99, 98, 97, 96, 95, 94, 86, 85, 84,
            83, 82, 81, 80, 79, 192, 191, 190, 189, 188, 187, 186, 185, 177, 176, 175, 174, 173, 172, 171, 170, 162,
            161, 160, 159, 158, 157, 156, 155, 147, 146, 145, 144, 143, 142, 141, 140, 132, 131, 130, 129, 128, 127,
            126, 125, 117, 116, 115, 114, 113, 112, 111, 110, 102, 101, 100, 99, 98, 97, 96, 95, 87, 86, 85, 84, 83, 82,
            81, 80, 193, 192, 191, 190, 189, 188, 187, 186, 178, 177, 176, 175, 174, 173, 172, 171, 163, 162, 161, 160,
            159, 158, 157, 156, 148, 147, 146, 145, 144, 143, 142, 141, 133, 132, 131, 130, 129, 128, 127, 126, 118,
            117, 116, 115, 114, 113, 112, 111, 103, 102, 101, 100, 99, 98, 97, 96, 88, 87, 86, 85, 84, 83, 82, 81, 194,
            193, 192, 191, 190, 189, 188, 187, 179, 178, 177, 176, 175, 174, 173, 172, 164, 163, 162, 161, 160, 159,
            158, 157, 149, 148, 147, 146, 145, 144, 143, 142, 134, 133, 132, 131, 130, 129, 128, 127, 119, 118, 117,
            116, 115, 114, 113, 112, 104, 103, 102, 101, 100, 99, 98, 97, 89, 88, 87, 86, 85, 84, 83, 82, 202, 201, 200,
            199, 198, 197, 196, 195, 187, 186, 185, 184, 183, 182, 181, 180, 172, 171, 170, 169, 168, 167, 166, 165,
            157, 156, 155, 154, 153, 152, 151, 150, 142, 141, 140, 139, 138, 137, 136, 135, 127, 126, 125, 124, 123,
            122, 121, 120, 112, 111, 110, 109, 108, 107, 106, 105, 97, 96, 95, 94, 93, 92, 91, 90, 203, 202, 201, 200,
            199, 198, 197, 196, 188, 187, 186, 185, 184, 183, 182, 181, 173, 172, 171, 170, 169, 168, 167, 166, 158,
            157, 156, 155, 154, 153, 152, 151, 143, 142, 141, 140, 139, 138, 137, 136, 128, 127, 126, 125, 124, 123,
            122, 121, 113, 112, 111, 110, 109, 108, 107, 106, 98, 97, 96, 95, 94, 93, 92, 91, 204, 203, 202, 201, 200,
            199, 198, 197, 189, 188, 187, 186, 185, 184, 183, 182, 174, 173, 172, 171, 170, 169, 168, 167, 159, 158,
            157, 156, 155, 154, 153, 152, 144, 143, 142, 141, 140, 139, 138, 137, 129, 128, 127, 126, 125, 124, 123,
            122, 114, 113, 112, 111, 110, 109, 108, 107, 99, 98, 97, 96, 95, 94, 93, 92, 205, 204, 203, 202, 201, 200,
            199, 198, 190, 189, 188, 187, 186, 185, 184, 183, 175, 174, 173, 172, 171, 170, 169, 168, 160, 159, 158,
            157, 156, 155, 154, 153, 145, 144, 143, 142, 141, 140, 139, 138, 130, 129, 128, 127, 126, 125, 124, 123,
            115, 114, 113, 112, 111, 110, 109, 108, 100, 99, 98, 97, 96, 95, 94, 93, 206, 205, 204, 203, 202, 201, 200,
            199, 191, 190, 189, 188, 187, 186, 185, 184, 176, 175, 174, 173, 172, 171, 170, 169, 161, 160, 159, 158,
            157, 156, 155, 154, 146, 145, 144, 143, 142, 141, 140, 139, 131, 130, 129, 128, 127, 126, 125, 124, 116,
            115, 114, 113, 112, 111, 110, 109, 101, 100, 99, 98, 97, 96, 95, 94, 207, 206, 205, 204, 203, 202, 201, 200,
            192, 191, 190, 189, 188, 187, 186, 185, 177, 176, 175, 174, 173, 172, 171, 170, 162, 161, 160, 159, 158,
            157, 156, 155, 147, 146, 145, 144, 143, 142, 141, 140, 132, 131, 130, 129, 128, 127, 126, 125, 117, 116,
            115, 114, 113, 112, 111, 110, 102, 101, 100, 99, 98, 97, 96, 95, 208, 207, 206, 205, 204, 203, 202, 201,
            193, 192, 191, 190, 189, 188, 187, 186, 178, 177, 176, 175, 174, 173, 172, 171, 163, 162, 161, 160, 159,
            158, 157, 156, 148, 147, 146, 145, 144, 143, 142, 141, 133, 132, 131, 130, 129, 128, 127, 126, 118, 117,
            116, 115, 114, 113, 112, 111, 103, 102, 101, 100, 99, 98, 97, 96, 209, 208, 207, 206, 205, 204, 203, 202,
            194, 193, 192, 191, 190, 189, 188, 187, 179, 178, 177, 176, 175, 174, 173, 172, 164, 163, 162, 161, 160,
            159, 158, 157, 149, 148, 147, 146, 145, 144, 143, 142, 134, 133, 132, 131, 130, 129, 128, 127, 119, 118,
            117, 116, 115, 114, 113, 112, 104, 103, 102, 101, 100, 99, 98, 97, 217, 216, 215, 214, 213, 212, 211, 210,
            202, 201, 200, 199, 198, 197, 196, 195, 187, 186, 185, 184, 183, 182, 181, 180, 172, 171, 170, 169, 168,
            167, 166, 165, 157, 156, 155, 154, 153, 152, 151, 150, 142, 141, 140, 139, 138, 137, 136, 135, 127, 126,
            125, 124, 123, 122, 121, 120, 112, 111, 110, 109, 108, 107, 106, 105, 218, 217, 216, 215, 214, 213, 212,
            211, 203, 202, 201, 200, 199, 198, 197, 196, 188, 187, 186, 185, 184, 183, 182, 181, 173, 172, 171, 170,
            169, 168, 167, 166, 158, 157, 156, 155, 154, 153, 152, 151, 143, 142, 141, 140, 139, 138, 137, 136, 128,
            127, 126, 125, 124, 123, 122, 121, 113, 112, 111, 110, 109, 108, 107, 106, 219, 218, 217, 216, 215, 214,
            213, 212, 204, 203, 202, 201, 200, 199, 198, 197, 189, 188, 187, 186, 185, 184, 183, 182, 174, 173, 172,
            171, 170, 169, 168, 167, 159, 158, 157, 156, 155, 154, 153, 152, 144, 143, 142, 141, 140, 139, 138, 137,
            129, 128, 127, 126, 125, 124, 123, 122, 114, 113, 112, 111, 110, 109, 108, 107, 220, 219, 218, 217, 216,
            215, 214, 213, 205, 204, 203, 202, 201, 200, 199, 198, 190, 189, 188, 187, 186, 185, 184, 183, 175, 174,
            173, 172, 171, 170, 169, 168, 160, 159, 158, 157, 156, 155, 154, 153, 145, 144, 143, 142, 141, 140, 139,
            138, 130, 129, 128, 127, 126, 125, 124, 123, 115, 114, 113, 112, 111, 110, 109, 108, 221, 220, 219, 218,
            217, 216, 215, 214, 206, 205, 204, 203, 202, 201, 200, 199, 191, 190, 189, 188, 187, 186, 185, 184, 176,
            175, 174, 173, 172, 171, 170, 169, 161, 160, 159, 158, 157, 156, 155, 154, 146, 145, 144, 143, 142, 141,
            140, 139, 131, 130, 129, 128, 127, 126, 125, 124, 116, 115, 114, 113, 112, 111, 110, 109, 222, 221, 220,
            219, 218, 217, 216, 215, 207, 206, 205, 204, 203, 202, 201, 200, 192, 191, 190, 189, 188, 187, 186, 185,
            177, 176, 175, 174, 173, 172, 171, 170, 162, 161, 160, 159, 158, 157, 156, 155, 147, 146, 145, 144, 143,
            142, 141, 140, 132, 131, 130, 129, 128, 127, 126, 125, 117, 116, 115, 114, 113, 112, 111, 110, 223, 222,
            221, 220, 219, 218, 217, 216, 208, 207, 206, 205, 204, 203, 202, 201, 193, 192, 191, 190, 189, 188, 187,
            186, 178, 177, 176, 175, 174, 173, 172, 171, 163, 162, 161, 160, 159, 158, 157, 156, 148, 147, 146, 145,
            144, 143, 142, 141, 133, 132, 131, 130, 129, 128, 127, 126, 118, 117, 116, 115, 114, 113, 112, 111, 224,
            223, 222, 221, 220, 219, 218, 217, 209, 208, 207, 206, 205, 204, 203, 202, 194, 193, 192, 191, 190, 189,
            188, 187, 179, 178, 177, 176, 175, 174, 173, 172, 164, 163, 162, 161, 160, 159, 158, 157, 149, 148, 147,
            146, 145, 144, 143, 142, 134, 133, 132, 131, 130, 129, 128, 127, 119, 118, 117, 116, 115, 114, 113, 112],
            'int32')

        layer = RelativeBias(8, 8, 8, 1)
        layer.build(None)

        result_table = self.evaluate(layer.rel_tab)
        self.assertAllClose(expected_table, result_table, atol=1e-3)

        result_index = self.evaluate(layer.rel_idx)
        self.assertAllClose(expected_index, result_index, atol=1e-3)

    def test_finetune(self):
        layer6 = RelativeBias(6, 6, 6, 1)
        layer6.build(None)
        expected = tf.gather(layer6.rel_tab, layer6.rel_idx)
        expected = tf.reshape(expected, [6] * 4 + [2])
        expected = self.evaluate(expected)

        layer10 = RelativeBias(10, 6, 10, 1)
        layer10.build(None)
        result = tf.gather(layer10.rel_tab, layer10.rel_idx)
        result = tf.reshape(result, [10] * 4 + [2])
        result = result[2:-2, 2:-2, 2:-2, 2:-2]
        result = self.evaluate(result)

        self.assertAllClose(expected, result)

    def test_halo(self):
        layer6 = RelativeBias(6, 6, 6, 1)
        layer6.build(None)
        expected = tf.gather(layer6.rel_tab, layer6.rel_idx)
        expected = tf.reshape(expected, [6] * 4 + [2])
        expected = self.evaluate(expected)

        layer12 = RelativeBias(6, 6, 12, 1)
        layer12.build(None)
        result = tf.gather(layer12.rel_tab, layer12.rel_idx)
        result = tf.reshape(result, [6] * 2 + [12] * 2 + [2])
        result = result[:, :, 3:-3, 3:-3]
        result = self.evaluate(result)

        self.assertAllClose(expected, result)


@test_combinations.run_all_keras_modes
class TestCHMSA(test_combinations.TestCase):
    def setUp(self):
        super(TestCHMSA, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestCHMSA, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            CHMSA,
            kwargs={'num_heads': 2, 'use_dw': False, 'qkv_bias': True, 'proj_bias': True},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            CHMSA,
            kwargs={'num_heads': 2, 'use_dw': True, 'qkv_bias': True, 'proj_bias': True},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            CHMSA,
            kwargs={'num_heads': 2, 'use_dw': False, 'qkv_bias': False, 'proj_bias': True},
            input_shape=[2, 16, 16, 4],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            CHMSA,
            kwargs={'num_heads': 4, 'use_dw': False, 'qkv_bias': True, 'proj_bias': False},
            input_shape=[2, 16, 16, 4],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()

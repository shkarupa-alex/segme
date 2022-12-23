import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from segme.policy.backbone.diy.coma.model import CoMA
from tensorflow.python.util import object_identity
from tensorflow.python.training.tracking import util as trackable_util


@test_combinations.run_all_keras_modes
class TestModel(test_combinations.TestCase):
    def setUp(self):
        super(TestModel, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestModel, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_drop_path(self):
        config = CoMA(
            embed_dim=64, stem_depth=2, stage_depths=(2, 2, 6, 4), weights=None, include_top=False,
            input_shape=(None, None, 3)).get_config()['layers']

        expected_drops = [
            ('stem_mbconv_0', 0.0), ('stem_mbconv_1', 0.010526315789473684),

            ('stage_0_reduce_mbconv', 0.0),
            ('stage_0_conv_0_mbconv', 0.021052631578947368),
            ('stage_0_conv_1_mbconv', 0.031578947368421054),
            ('stage_0_attn_2_drop1', 0.042105263157894736), ('stage_0_attn_2_drop2', 0.042105263157894736),

            ('stage_1_reduce_mbconv', 0.0),
            ('stage_1_conv_0_mbconv', 0.05263157894736842),
            ('stage_1_conv_1_mbconv', 0.06315789473684211),
            ('stage_1_attn_2_drop1', 0.07368421052631578), ('stage_1_attn_2_drop2', 0.07368421052631578),

            ('stage_2_reduce_mbconv', 0.0),
            ('stage_2_attn_0_drop1', 0.08421052631578947), ('stage_2_attn_0_drop2', 0.08421052631578947),
            ('stage_2_attn_1_drop1', 0.09473684210526316), ('stage_2_attn_1_drop2', 0.09473684210526316),
            ('stage_2_attn_2_drop1', 0.10526315789473684), ('stage_2_attn_2_drop2', 0.10526315789473684),
            ('stage_2_attn_3_drop1', 0.11578947368421053), ('stage_2_attn_3_drop2', 0.11578947368421053),
            ('stage_2_attn_4_drop1', 0.12631578947368421), ('stage_2_attn_4_drop2', 0.12631578947368421),
            ('stage_2_attn_5_drop1', 0.1368421052631579), ('stage_2_attn_5_drop2', 0.1368421052631579),
            ('stage_2_attn_6_drop1', 0.14736842105263157), ('stage_2_attn_6_drop2', 0.14736842105263157),

            ('stage_3_reduce_mbconv', 0.0),
            ('stage_3_attn_0_drop1', 0.15789473684210525), ('stage_3_attn_0_drop2', 0.15789473684210525),
            ('stage_3_attn_1_drop1', 0.16842105263157894), ('stage_3_attn_1_drop2', 0.16842105263157894),
            ('stage_3_attn_2_drop1', 0.17894736842105263), ('stage_3_attn_2_drop2', 0.17894736842105263),
            ('stage_3_attn_3_drop1', 0.18947368421052632), ('stage_3_attn_3_drop2', 0.18947368421052632),
            ('stage_3_attn_4_drop1', 0.2), ('stage_3_attn_4_drop2', 0.2)]

        actual_drops = []
        for layer in config:
            if 'SegMe>Common>DropPath' == layer['class_name']:
                actual_drops.append((layer['config']['name'], layer['config']['rate']))
            if 'SegMe>Common>MBConv' == layer['class_name']:
                actual_drops.append((layer['config']['name'], layer['config']['drop_ratio']))
        self.assertListEqual(expected_drops, actual_drops)

    def test_residual_gamma(self):
        config = CoMA(
            embed_dim=64, stem_depth=2, stage_depths=(2, 2, 6, 4), weights=None, include_top=False,
            input_shape=(None, None, 3)).get_config()['layers']

        expected_gammas = [
            ('stem_mbconv_0', 0.1),
            ('stem_mbconv_1', 0.09565652173913045),

            ('stage_0_reduce_mbconv', 0.09131304347826087),
            ('stage_0_conv_0_mbconv', 0.08696956521739131),
            ('stage_0_conv_1_mbconv', 0.08262608695652174),
            ('stage_0_attn_2_norm1', 0.07828260869565218), ('stage_0_attn_2_norm2', 0.07828260869565218),

            ('stage_1_reduce_mbconv', 0.07393913043478262),
            ('stage_1_conv_0_mbconv', 0.06959565217391306),
            ('stage_1_conv_1_mbconv', 0.06525217391304348),
            ('stage_1_attn_2_norm1', 0.06090869565217392), ('stage_1_attn_2_norm2', 0.06090869565217392),

            ('stage_2_reduce_mbconv', 0.05656521739130435),
            ('stage_2_attn_0_norm1', 0.05222173913043479), ('stage_2_attn_0_norm2', 0.05222173913043479),
            ('stage_2_attn_1_norm1', 0.047878260869565224), ('stage_2_attn_1_norm2', 0.047878260869565224),
            ('stage_2_attn_2_norm1', 0.04353478260869566), ('stage_2_attn_2_norm2', 0.04353478260869566),
            ('stage_2_attn_3_norm1', 0.0391913043478261), ('stage_2_attn_3_norm2', 0.0391913043478261),
            ('stage_2_attn_4_norm1', 0.034847826086956524), ('stage_2_attn_4_norm2', 0.034847826086956524),
            ('stage_2_attn_5_norm1', 0.030504347826086964), ('stage_2_attn_5_norm2', 0.030504347826086964),
            ('stage_2_attn_6_norm1', 0.026160869565217404), ('stage_2_attn_6_norm2', 0.026160869565217404),

            ('stage_3_reduce_mbconv', 0.02181739130434783),
            ('stage_3_attn_0_norm1', 0.01747391304347827), ('stage_3_attn_0_norm2', 0.01747391304347827),
            ('stage_3_attn_1_norm1', 0.013130434782608696), ('stage_3_attn_1_norm2', 0.013130434782608696),
            ('stage_3_attn_2_norm1', 0.008786956521739137), ('stage_3_attn_2_norm2', 0.008786956521739137),
            ('stage_3_attn_3_norm1', 0.004443478260869577), ('stage_3_attn_3_norm2', 0.004443478260869577),
            ('stage_3_attn_4_norm1', 0.0001), ('stage_3_attn_4_norm2', 0.0001)]

        actual_gammas = []
        for layer in config:
            if 'gamma_initializer' in layer['config'] and \
                    'Constant' == layer['config']['gamma_initializer']['class_name']:
                actual_gammas.append((layer['config']['name'], layer['config']['gamma_initializer']['config']['value']))
        self.assertListEqual(expected_gammas, actual_gammas)

    @parameterized.parameters((112,), (384,), (512,), (576,))
    def test_attn_dil2(self, size):
        config = CoMA(
            embed_dim=64, stem_depth=2, stage_depths=(2, 2, 6, 4), pretrain_size=size, weights=None, include_top=False,
            input_shape=(None, None, 3)).get_config()['layers']

        expected_atentions = {
            112: [
                ('CHMSA', 'stage_0_attn_2_channel', 2),

                ('CHMSA', 'stage_1_attn_2_channel', 4),

                ('DHMSA', 'stage_2_attn_0_window', 8, 1), ('DHMSA', 'stage_2_attn_1_window', 8, 1),
                ('DHMSA', 'stage_2_attn_2_window', 8, 1), ('DHMSA', 'stage_2_attn_3_window', 8, 1),
                ('DHMSA', 'stage_2_attn_4_window', 8, 1), ('DHMSA', 'stage_2_attn_5_window', 8, 1),
                ('CHMSA', 'stage_2_attn_6_channel', 8),

                ('DHMSA', 'stage_3_attn_0_window', 16, 1), ('DHMSA', 'stage_3_attn_1_window', 16, 1),
                ('DHMSA', 'stage_3_attn_2_window', 16, 1), ('DHMSA', 'stage_3_attn_3_window', 16, 1),
                ('CHMSA', 'stage_3_attn_4_channel', 16)],
            384: [
                ('CHMSA', 'stage_0_attn_2_channel', 2),

                ('CHMSA', 'stage_1_attn_2_channel', 4),

                ('DHMSA', 'stage_2_attn_0_window', 8, 1), ('DHMSA', 'stage_2_attn_1_window', 8, 2),
                ('DHMSA', 'stage_2_attn_2_window', 8, 1), ('DHMSA', 'stage_2_attn_3_window', 8, 2),
                ('DHMSA', 'stage_2_attn_4_window', 8, 1), ('DHMSA', 'stage_2_attn_5_window', 8, 2),
                ('CHMSA', 'stage_2_attn_6_channel', 8),

                ('DHMSA', 'stage_3_attn_0_window', 16, 1), ('DHMSA', 'stage_3_attn_1_window', 16, 1),
                ('DHMSA', 'stage_3_attn_2_window', 16, 1), ('DHMSA', 'stage_3_attn_3_window', 16, 1),
                ('CHMSA', 'stage_3_attn_4_channel', 16)],
            512: [
                ('CHMSA', 'stage_0_attn_2_channel', 2),

                ('CHMSA', 'stage_1_attn_2_channel', 4),

                ('DHMSA', 'stage_2_attn_0_window', 8, 1), ('DHMSA', 'stage_2_attn_1_window', 8, 2),
                ('DHMSA', 'stage_2_attn_2_window', 8, 1), ('DHMSA', 'stage_2_attn_3_window', 8, 2),
                ('DHMSA', 'stage_2_attn_4_window', 8, 1), ('DHMSA', 'stage_2_attn_5_window', 8, 2),
                ('CHMSA', 'stage_2_attn_6_channel', 8),

                ('DHMSA', 'stage_3_attn_0_window', 16, 1), ('DHMSA', 'stage_3_attn_1_window', 16, 1),
                ('DHMSA', 'stage_3_attn_2_window', 16, 1), ('DHMSA', 'stage_3_attn_3_window', 16, 1),
                ('CHMSA', 'stage_3_attn_4_channel', 16)],
            576: [
                ('CHMSA', 'stage_0_attn_2_channel', 2),

                ('CHMSA', 'stage_1_attn_2_channel', 4),

                ('DHMSA', 'stage_2_attn_0_window', 8, 1), ('DHMSA', 'stage_2_attn_1_window', 8, 2),
                ('DHMSA', 'stage_2_attn_2_window', 8, 1), ('DHMSA', 'stage_2_attn_3_window', 8, 3),
                ('DHMSA', 'stage_2_attn_4_window', 8, 1), ('DHMSA', 'stage_2_attn_5_window', 8, 2),
                ('CHMSA', 'stage_2_attn_6_channel', 8),

                ('DHMSA', 'stage_3_attn_0_window', 16, 1), ('DHMSA', 'stage_3_attn_1_window', 16, 1),
                ('DHMSA', 'stage_3_attn_2_window', 16, 1), ('DHMSA', 'stage_3_attn_3_window', 16, 1),
                ('CHMSA', 'stage_3_attn_4_channel', 16)]
        }

        actual_attentions = []
        for layer in config:
            if 'SegMe>Policy>Backbone>DIY>CoMA>DHMSA' == layer['class_name']:
                actual_attentions.append(
                    ('DHMSA', layer['config']['name'], layer['config']['num_heads'], layer['config']['dilation_rate']))
            if 'SegMe>Policy>Backbone>DIY>CoMA>GGMSA' == layer['class_name']:
                actual_attentions.append(
                    ('GGMSA', layer['config']['name'], layer['config']['num_heads']))
            if 'SegMe>Policy>Backbone>DIY>CoMA>CHMSA' == layer['class_name']:
                actual_attentions.append(
                    ('CHMSA', layer['config']['name'], layer['config']['num_heads']))
        self.assertListEqual(expected_atentions[size], actual_attentions)

    @parameterized.parameters((False,), (True,))
    def test_train(self, use_fp16):
        if use_fp16:
            mixed_precision.set_global_policy('mixed_float16')

        model = CoMA(embed_dim=64, stem_depth=2, stage_depths=(4, 4, 4, 4), weights=None)
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=test_utils.should_run_eagerly())

        images = np.random.random((10, 384, 384, 3)).astype('float32')
        labels = (np.random.random((10, 1)) + 0.5).astype('int64')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)

    def test_finite(self):
        model = CoMA(
            embed_dim=64, stem_depth=2, stage_depths=(4, 4, 4, 4), weights=None, include_top=False,
            input_shape=(None, None, 3))
        outputs = model(np.random.uniform(0., 255., [2, 384, 384, 3]).astype('float32'))
        outputs = self.evaluate(outputs)
        self.assertTrue(np.isfinite(outputs).all())

    def test_var_shape(self):
        model = CoMA(
            embed_dim=64, stem_depth=2, stage_depths=(4, 4, 4, 4), weights=None, include_top=False,
            input_shape=(None, None, 3))
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=test_utils.should_run_eagerly())

        images = np.random.random((10, 512, 384, 3)).astype('float32')
        labels = (np.random.random((10, 16, 12, 512)) + 0.5).astype('int64')
        model.fit(images, labels, epochs=1, batch_size=2)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)


if __name__ == '__main__':
    tf.test.main()

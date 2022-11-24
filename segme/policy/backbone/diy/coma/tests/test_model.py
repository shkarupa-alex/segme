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
            ('stem_mbconv_0', 0.0), ('stem_mbconv_1', 0.0),

            ('stage_0_reduce_mbconv', 0.0), ('stage_0_reduce_drop', 0.0),

            ('stage_0_conv_0_mbconv', 0.0),
            ('stage_0_conv_1_mbconv', 0.011764705882352941),
            ('stage_0_attn_2_drop1', 0.023529411764705882), ('stage_0_attn_2_drop2', 0.023529411764705882),

            ('stage_1_reduce_mbconv', 0.0), ('stage_1_reduce_drop', 0.0),

            ('stage_1_conv_0_mbconv', 0.03529411764705882),
            ('stage_1_conv_1_mbconv', 0.047058823529411764),
            ('stage_1_attn_2_drop1', 0.058823529411764705), ('stage_1_attn_2_drop2', 0.058823529411764705),

            ('stage_2_reduce_mbconv', 0.0), ('stage_2_reduce_drop', 0.0),

            ('stage_2_attn_0_drop1', 0.07058823529411765), ('stage_2_attn_0_drop2', 0.07058823529411765),
            ('stage_2_attn_1_drop1', 0.08235294117647059), ('stage_2_attn_1_drop2', 0.08235294117647059),
            ('stage_2_attn_2_drop1', 0.09411764705882353), ('stage_2_attn_2_drop2', 0.09411764705882353),
            ('stage_2_attn_3_drop1', 0.10588235294117647), ('stage_2_attn_3_drop2', 0.10588235294117647),
            ('stage_2_attn_4_drop1', 0.11764705882352941), ('stage_2_attn_4_drop2', 0.11764705882352941),
            ('stage_2_attn_5_drop1', 0.12941176470588234), ('stage_2_attn_5_drop2', 0.12941176470588234),
            ('stage_2_attn_6_drop1', 0.1411764705882353), ('stage_2_attn_6_drop2', 0.1411764705882353),

            ('stage_3_reduce_mbconv', 0.0), ('stage_3_reduce_drop', 0.0),

            ('stage_3_attn_0_drop1', 0.15294117647058825), ('stage_3_attn_0_drop2', 0.15294117647058825),
            ('stage_3_attn_1_drop1', 0.16470588235294117), ('stage_3_attn_1_drop2', 0.16470588235294117),
            ('stage_3_attn_2_drop1', 0.1764705882352941), ('stage_3_attn_2_drop2', 0.1764705882352941),
            ('stage_3_attn_3_drop1', 0.18823529411764706), ('stage_3_attn_3_drop2', 0.18823529411764706),
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
            ('stem_mbconv_0', 0.1), ('stem_mbconv_1', 0.1),

            ('stage_0_reduce_mbconv', 0.09545909090909091),

            ('stage_0_conv_0_mbconv', 0.09091818181818183),
            ('stage_0_conv_1_mbconv', 0.08637727272727273),
            ('stage_0_attn_2_norm1', 0.08183636363636364), ('stage_0_attn_2_norm2', 0.08183636363636364),

            ('stage_1_reduce_mbconv', 0.07729545454545456),

            ('stage_1_conv_0_mbconv', 0.07275454545454546),
            ('stage_1_conv_1_mbconv', 0.06821363636363637),
            ('stage_1_attn_2_norm1', 0.06367272727272727), ('stage_1_attn_2_norm2', 0.06367272727272727),

            ('stage_2_reduce_mbconv', 0.05913181818181819), ('stage_2_attn_0_norm1', 0.05459090909090909),
            ('stage_2_attn_0_norm2', 0.05459090909090909), ('stage_2_attn_1_norm1', 0.050050000000000004),
            ('stage_2_attn_1_norm2', 0.050050000000000004), ('stage_2_attn_2_norm1', 0.045509090909090916),
            ('stage_2_attn_2_norm2', 0.045509090909090916), ('stage_2_attn_3_norm1', 0.04096818181818182),
            ('stage_2_attn_3_norm2', 0.04096818181818182), ('stage_2_attn_4_norm1', 0.036427272727272725),
            ('stage_2_attn_4_norm2', 0.036427272727272725), ('stage_2_attn_5_norm1', 0.03188636363636364),
            ('stage_2_attn_5_norm2', 0.03188636363636364), ('stage_2_attn_6_norm1', 0.027345454545454548),
            ('stage_2_attn_6_norm2', 0.027345454545454548),

            ('stage_3_reduce_mbconv', 0.022804545454545452),

            ('stage_3_attn_0_norm1', 0.01826363636363637), ('stage_3_attn_0_norm2', 0.01826363636363637),
            ('stage_3_attn_1_norm1', 0.013722727272727275), ('stage_3_attn_1_norm2', 0.013722727272727275),
            ('stage_3_attn_2_norm1', 0.00918181818181818), ('stage_3_attn_2_norm2', 0.00918181818181818),
            ('stage_3_attn_3_norm1', 0.004640909090909098), ('stage_3_attn_3_norm2', 0.004640909090909098),
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

                ('DHMSA', 'stage_3_attn_0_window', 16, 1), ('GGMSA', 'stage_3_attn_1_grid', 16),
                ('DHMSA', 'stage_3_attn_2_window', 16, 1), ('GGMSA', 'stage_3_attn_3_grid', 16),
                ('CHMSA', 'stage_3_attn_4_channel', 16)],
            384: [
                ('CHMSA', 'stage_0_attn_2_channel', 2),

                ('CHMSA', 'stage_1_attn_2_channel', 4),

                ('DHMSA', 'stage_2_attn_0_window', 8, 1), ('DHMSA', 'stage_2_attn_1_window', 8, 2),
                ('DHMSA', 'stage_2_attn_2_window', 8, 1), ('DHMSA', 'stage_2_attn_3_window', 8, 2),
                ('DHMSA', 'stage_2_attn_4_window', 8, 1), ('DHMSA', 'stage_2_attn_5_window', 8, 2),
                ('CHMSA', 'stage_2_attn_6_channel', 8),

                ('DHMSA', 'stage_3_attn_0_window', 16, 1), ('GGMSA', 'stage_3_attn_1_grid', 16),
                ('DHMSA', 'stage_3_attn_2_window', 16, 1), ('GGMSA', 'stage_3_attn_3_grid', 16),
                ('CHMSA', 'stage_3_attn_4_channel', 16)],
            512: [
                ('CHMSA', 'stage_0_attn_2_channel', 2),

                ('CHMSA', 'stage_1_attn_2_channel', 4),

                ('DHMSA', 'stage_2_attn_0_window', 8, 1), ('DHMSA', 'stage_2_attn_1_window', 8, 2),
                ('DHMSA', 'stage_2_attn_2_window', 8, 1), ('DHMSA', 'stage_2_attn_3_window', 8, 2),
                ('DHMSA', 'stage_2_attn_4_window', 8, 1), ('DHMSA', 'stage_2_attn_5_window', 8, 2),
                ('CHMSA', 'stage_2_attn_6_channel', 8),

                ('DHMSA', 'stage_3_attn_0_window', 16, 1), ('GGMSA', 'stage_3_attn_1_grid', 16),
                ('DHMSA', 'stage_3_attn_2_window', 16, 1), ('GGMSA', 'stage_3_attn_3_grid', 16),
                ('CHMSA', 'stage_3_attn_4_channel', 16)],
            576: [
                ('CHMSA', 'stage_0_attn_2_channel', 2),

                ('CHMSA', 'stage_1_attn_2_channel', 4),

                ('DHMSA', 'stage_2_attn_0_window', 8, 1), ('DHMSA', 'stage_2_attn_1_window', 8, 2),
                ('DHMSA', 'stage_2_attn_2_window', 8, 1), ('DHMSA', 'stage_2_attn_3_window', 8, 3),
                ('DHMSA', 'stage_2_attn_4_window', 8, 1), ('DHMSA', 'stage_2_attn_5_window', 8, 2),
                ('CHMSA', 'stage_2_attn_6_channel', 8),

                ('DHMSA', 'stage_3_attn_0_window', 16, 1), ('GGMSA', 'stage_3_attn_1_grid', 16),
                ('DHMSA', 'stage_3_attn_2_window', 16, 1), ('GGMSA', 'stage_3_attn_3_grid', 16),
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

    # @parameterized.parameters((False,), (True,))
    # def test_train(self, use_fp16):
    #     if use_fp16:
    #         mixed_precision.set_global_policy('mixed_float16')
    #
    #     model = CoMA(embed_dim=64, stem_depth=2, stage_depths=(4, 4, 4, 4), weights=None)
    #     model.compile(optimizer='rmsprop', loss='mse', run_eagerly=test_utils.should_run_eagerly())
    #
    #     images = np.random.random((10, 384, 384, 3)).astype('float32')
    #     labels = (np.random.random((10, 1)) + 0.5).astype('int32')
    #     model.fit(images, labels, epochs=1, batch_size=2)
    #
    #     # test config
    #     model.get_config()
    #
    #     # check whether the model variables are present in the trackable list of objects
    #     checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
    #     for v in model.variables:
    #         self.assertIn(v, checkpointed_objects)
    #
    # def test_finite(self):
    #     model = CoMA(
    #         embed_dim=64, stem_depth=2, stage_depths=(4, 4, 4, 4), weights=None, include_top=False,
    #         input_shape=(None, None, 3))
    #     outputs = model(np.random.uniform(0., 255., [2, 384, 384, 3]).astype('float32'))
    #     outputs = self.evaluate(outputs)
    #     self.assertTrue(np.isfinite(outputs).all())
    #
    # def test_var_shape(self):
    #     model = CoMA(
    #         embed_dim=64, stem_depth=2, stage_depths=(4, 4, 4, 4), weights=None, include_top=False,
    #         input_shape=(None, None, 3))
    #     model.compile(optimizer='rmsprop', loss='mse', run_eagerly=test_utils.should_run_eagerly())
    #
    #     images = np.random.random((10, 512, 384, 3)).astype('float32')
    #     labels = (np.random.random((10, 16, 12, 512)) + 0.5).astype('int32')
    #     model.fit(images, labels, epochs=1, batch_size=2)
    #
    #     # test config
    #     model.get_config()
    #
    #     # check whether the model variables are present in the trackable list of objects
    #     checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
    #     for v in model.variables:
    #         self.assertIn(v, checkpointed_objects)

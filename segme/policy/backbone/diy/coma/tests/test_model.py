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

    @staticmethod
    def _values_from_config(cfg, cls, prop):
        if isinstance(cls, str):
            cls = [cls]

        values = []
        for layer in cfg['layers']:
            if layer['class_name'] not in cls:
                continue

            if prop not in layer['config']:
                continue

            values.append((layer['name'], layer['config'][prop]))

        return values

    def test_drop_path(self):
        config = CoMA(
            embed_dim=64, stem_depth=2, stage_depths=(2, 2, 6, 4), weights=None, include_top=False,
            input_shape=(None, None, 3)).get_config()

        expected_drops = [
            ('stem_1_drop', 0.0), ('stem_2_drop', 0.010526315789473684),

            ('stage_0_conv_0_mlpconv_drop', 0.021052631578947368),
            ('stage_0_conv_1_mlpconv_drop', 0.031578947368421054),
            ('stage_0_attn_2_chmsa_drop', 0.042105263157894736), ('stage_0_attn_2_mlpconv_drop', 0.042105263157894736),

            ('stage_1_conv_0_mlpconv_drop', 0.05263157894736842),
            ('stage_1_conv_1_mlpconv_drop', 0.06315789473684211),
            ('stage_1_attn_2_chmsa_drop', 0.07368421052631578), ('stage_1_attn_2_mlpconv_drop', 0.07368421052631578),

            ('stage_2_attn_0_dhmsa_drop', 0.08421052631578947), ('stage_2_attn_0_mlpconv_drop', 0.08421052631578947),
            ('stage_2_attn_1_dhmsa_drop', 0.09473684210526316), ('stage_2_attn_1_mlpconv_drop', 0.09473684210526316),
            ('stage_2_attn_2_dhmsa_drop', 0.10526315789473684), ('stage_2_attn_2_mlpconv_drop', 0.10526315789473684),
            ('stage_2_attn_3_dhmsa_drop', 0.11578947368421053), ('stage_2_attn_3_mlpconv_drop', 0.11578947368421053),
            ('stage_2_attn_4_dhmsa_drop', 0.12631578947368421), ('stage_2_attn_4_mlpconv_drop', 0.12631578947368421),
            ('stage_2_attn_5_dhmsa_drop', 0.1368421052631579), ('stage_2_attn_5_mlpconv_drop', 0.1368421052631579),
            ('stage_2_attn_6_chmsa_drop', 0.14736842105263157), ('stage_2_attn_6_mlpconv_drop', 0.14736842105263157),

            ('stage_3_attn_0_dhmsa_drop', 0.15789473684210525), ('stage_3_attn_0_mlpconv_drop', 0.15789473684210525),
            ('stage_3_attn_1_dhmsa_drop', 0.16842105263157894), ('stage_3_attn_1_mlpconv_drop', 0.16842105263157894),
            ('stage_3_attn_2_dhmsa_drop', 0.17894736842105263), ('stage_3_attn_2_mlpconv_drop', 0.17894736842105263),
            ('stage_3_attn_3_dhmsa_drop', 0.18947368421052632), ('stage_3_attn_3_mlpconv_drop', 0.18947368421052632),
            ('stage_3_attn_4_chmsa_drop', 0.2), ('stage_3_attn_4_mlpconv_drop', 0.2)]

        actual_drops = TestModel._values_from_config(config, 'SegMe>Common>DropPath', 'rate')
        self.assertListEqual(expected_drops, actual_drops)

    def test_residual_gamma(self):
        config = CoMA(
            embed_dim=64, stem_depth=2, stage_depths=(2, 2, 6, 4), weights=None, include_top=False,
            input_shape=(None, None, 3)).get_config()

        expected_gammas = [
            ('stem_1_norm', 0.01),
            ('stem_2_norm', 0.00947421052631579),

            ('stage_0_conv_0_mlpconv_norm', 0.00894842105263158),
            ('stage_0_conv_1_mlpconv_norm', 0.008422631578947369),
            ('stage_0_attn_2_chmsa_norm', 0.007896842105263157), ('stage_0_attn_2_mlpconv_norm', 0.007896842105263157),

            ('stage_1_conv_0_mlpconv_norm', 0.0073710526315789475),
            ('stage_1_conv_1_mlpconv_norm', 0.006845263157894736),
            ('stage_1_attn_2_chmsa_norm', 0.006319473684210526), ('stage_1_attn_2_mlpconv_norm', 0.006319473684210526),

            ('stage_2_attn_0_dhmsa_norm', 0.0057936842105263155),
            ('stage_2_attn_0_mlpconv_norm', 0.0057936842105263155),
            ('stage_2_attn_1_dhmsa_norm', 0.005267894736842105), ('stage_2_attn_1_mlpconv_norm', 0.005267894736842105),
            ('stage_2_attn_2_dhmsa_norm', 0.004742105263157895), ('stage_2_attn_2_mlpconv_norm', 0.004742105263157895),
            ('stage_2_attn_3_dhmsa_norm', 0.0042163157894736835),
            ('stage_2_attn_3_mlpconv_norm', 0.0042163157894736835),
            ('stage_2_attn_4_dhmsa_norm', 0.003690526315789473), ('stage_2_attn_4_mlpconv_norm', 0.003690526315789473),
            ('stage_2_attn_5_dhmsa_norm', 0.0031647368421052627),
            ('stage_2_attn_5_mlpconv_norm', 0.0031647368421052627),
            ('stage_2_attn_6_chmsa_norm', 0.0026389473684210515),
            ('stage_2_attn_6_mlpconv_norm', 0.0026389473684210515),

            ('stage_3_attn_0_dhmsa_norm', 0.002113157894736841), ('stage_3_attn_0_mlpconv_norm', 0.002113157894736841),
            ('stage_3_attn_1_dhmsa_norm', 0.0015873684210526307),
            ('stage_3_attn_1_mlpconv_norm', 0.0015873684210526307),
            ('stage_3_attn_2_dhmsa_norm', 0.0010615789473684203),
            ('stage_3_attn_2_mlpconv_norm', 0.0010615789473684203),
            ('stage_3_attn_3_dhmsa_norm', 0.00053578947368421), ('stage_3_attn_3_mlpconv_norm', 0.00053578947368421),
            ('stage_3_attn_4_chmsa_norm', 1e-05), ('stage_3_attn_4_mlpconv_norm', 1e-05)]

        actual_gammas = TestModel._values_from_config(
            config, 'SegMe>Policy>Normalization>BatchNorm', 'gamma_initializer')
        actual_gammas = [(ag[0], ag[1]['config']['value']) for ag in actual_gammas if 'Constant' == ag[1]['class_name']]
        self.assertListEqual(expected_gammas, actual_gammas)

    @parameterized.parameters((112,), (384,), (512,), (576,))
    def test_attention_dilation(self, size):
        config = CoMA(
            embed_dim=64, stem_depth=2, stage_depths=(2, 2, 6, 4), pretrain_size=size, weights=None, include_top=False,
            input_shape=(None, None, 3)).get_config()

        expected_dilations = {
            112: [
                ('stage_2_attn_0_dhmsa_attn', 1), ('stage_2_attn_1_dhmsa_attn', 1), ('stage_2_attn_2_dhmsa_attn', 1),
                ('stage_2_attn_3_dhmsa_attn', 1), ('stage_2_attn_4_dhmsa_attn', 1), ('stage_2_attn_5_dhmsa_attn', 1),
                ('stage_3_attn_0_dhmsa_attn', 1), ('stage_3_attn_1_dhmsa_attn', 1),

                ('stage_3_attn_2_dhmsa_attn', 1), ('stage_3_attn_3_dhmsa_attn', 1)],
            384: [
                ('stage_2_attn_0_dhmsa_attn', 1), ('stage_2_attn_1_dhmsa_attn', 2),
                ('stage_2_attn_2_dhmsa_attn', 1), ('stage_2_attn_3_dhmsa_attn', 2),
                ('stage_2_attn_4_dhmsa_attn', 1), ('stage_2_attn_5_dhmsa_attn', 2),

                ('stage_3_attn_0_dhmsa_attn', 1), ('stage_3_attn_1_dhmsa_attn', 1),
                ('stage_3_attn_2_dhmsa_attn', 1), ('stage_3_attn_3_dhmsa_attn', 1)],
            512: [
                ('stage_2_attn_0_dhmsa_attn', 1), ('stage_2_attn_1_dhmsa_attn', 2),
                ('stage_2_attn_2_dhmsa_attn', 1), ('stage_2_attn_3_dhmsa_attn', 2),
                ('stage_2_attn_4_dhmsa_attn', 1), ('stage_2_attn_5_dhmsa_attn', 2),

                ('stage_3_attn_0_dhmsa_attn', 1), ('stage_3_attn_1_dhmsa_attn', 1),
                ('stage_3_attn_2_dhmsa_attn', 1), ('stage_3_attn_3_dhmsa_attn', 1)],
            576: [
                ('stage_2_attn_0_dhmsa_attn', 1), ('stage_2_attn_1_dhmsa_attn', 2),
                ('stage_2_attn_2_dhmsa_attn', 1), ('stage_2_attn_3_dhmsa_attn', 3),
                ('stage_2_attn_4_dhmsa_attn', 1), ('stage_2_attn_5_dhmsa_attn', 2),

                ('stage_3_attn_0_dhmsa_attn', 1), ('stage_3_attn_1_dhmsa_attn', 1),
                ('stage_3_attn_2_dhmsa_attn', 1), ('stage_3_attn_3_dhmsa_attn', 1)]
        }

        actual_dilations = TestModel._values_from_config(
            config, 'SegMe>Policy>Backbone>DIY>CoMA>DHMSA', 'dilation_rate')
        self.assertListEqual(expected_dilations[size], actual_dilations)

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
        run_eagerly = test_utils.should_run_eagerly()
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=run_eagerly, jit_compile=not run_eagerly)

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

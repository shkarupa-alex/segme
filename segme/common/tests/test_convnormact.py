import tensorflow as tf
from keras import initializers, layers
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.common.convnormact import Conv, Norm, Act, ConvNormAct
from segme.policy import cnapol, sameconv, norm, act


@test_combinations.run_all_keras_modes
class TestConv(test_combinations.TestCase):
    def setUp(self):
        super(TestConv, self).setUp()
        self.default_convnormact = cnapol.global_policy()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestConv, self).tearDown()
        cnapol.set_global_policy(self.default_convnormact)
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            Conv,
            kwargs={'filters': 4, 'kernel_size': 3},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            Conv,
            kwargs={'filters': None, 'kernel_size': 3},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float32'
        )

        with cnapol.policy_scope('stdconv-gn-leakyrelu'):
            test_utils.layer_test(
                Conv,
                kwargs={'filters': 4, 'kernel_size': 3},
                input_shape=[2, 16, 16, 3],
                input_dtype='float32',
                expected_output_shape=[None, 16, 16, 4],
                expected_output_dtype='float32'
            )
            test_utils.layer_test(
                Conv,
                kwargs={'filters': None, 'kernel_size': 3},
                input_shape=[2, 16, 16, 3],
                input_dtype='float32',
                expected_output_shape=[None, 16, 16, 3],
                expected_output_dtype='float32'
            )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            Conv,
            kwargs={'filters': 4, 'kernel_size': 3},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float16'
        )
        test_utils.layer_test(
            Conv,
            kwargs={'filters': None, 'kernel_size': 3},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float16'
        )

        with cnapol.policy_scope('stdconv-gn-leakyrelu'):
            test_utils.layer_test(
                Conv,
                kwargs={'filters': 4, 'kernel_size': 3},
                input_shape=[2, 16, 16, 3],
                input_dtype='float16',
                expected_output_shape=[None, 16, 16, 4],
                expected_output_dtype='float16'
            )
            test_utils.layer_test(
                Conv,
                kwargs={'filters': None, 'kernel_size': 3},
                input_shape=[2, 16, 16, 3],
                input_dtype='float16',
                expected_output_shape=[None, 16, 16, 3],
                expected_output_dtype='float16'
            )

    def test_conv_bn_relu(self):
        convinst = Conv(4, 3)
        convinst.build([None, None, None, 3])

        self.assertIsInstance(convinst.conv, sameconv.SameConv)
        self.assertEqual(convinst.conv.filters, 4)
        self.assertTupleEqual(convinst.conv.kernel_size, (3, 3))

    def test_dwconv_bn_relu(self):
        convinst = Conv(None, 3)
        convinst.build([None, None, None, 3])

        self.assertIsInstance(convinst.conv, sameconv.SameDepthwiseConv)
        self.assertTupleEqual(convinst.conv.kernel_size, (3, 3))

    def test_policy_scope_memorize(self):
        with cnapol.policy_scope('stdconv-gn-leakyrelu'):
            convinst = Conv(4, 3)
        convinst.build([None, None, None, 3])

        self.assertIsInstance(convinst.conv, sameconv.SameStandardizedConv)
        self.assertEqual(convinst.conv.filters, 4)
        self.assertTupleEqual(convinst.conv.kernel_size, (3, 3))

        restored = Conv.from_config(convinst.get_config())
        restored.build([None, None, None, 3])

        self.assertIsInstance(restored.conv, sameconv.SameStandardizedConv)
        self.assertEqual(restored.conv.filters, 4)
        self.assertTupleEqual(restored.conv.kernel_size, (3, 3))

    def test_policy_override_kwargs(self):
        with cnapol.policy_scope('stdconv-gn-leakyrelu'):
            convinst = Conv(4, 3, strides=2)
        convinst.build([None, None, None, 3])

        restored = Conv.from_config(convinst.get_config())
        restored.build([None, None, None, 3])

        self.assertIsInstance(restored.conv, sameconv.SameStandardizedConv)
        self.assertTupleEqual(restored.conv.strides, (2, 2))


@test_combinations.run_all_keras_modes
class TestNorm(test_combinations.TestCase):
    def setUp(self):
        super(TestNorm, self).setUp()
        self.default_convnormact = cnapol.global_policy()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestNorm, self).tearDown()
        cnapol.set_global_policy(self.default_convnormact)
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            Norm,
            kwargs={},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float32'
        )

        with cnapol.policy_scope('stdconv-gn-leakyrelu'):
            test_utils.layer_test(
                Norm,
                kwargs={},
                input_shape=[2, 16, 16, 3],
                input_dtype='float32',
                expected_output_shape=[None, 16, 16, 3],
                expected_output_dtype='float32'
            )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            Norm,
            kwargs={},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float16'
        )

        with cnapol.policy_scope('stdconv-gn-leakyrelu'):
            test_utils.layer_test(
                Norm,
                kwargs={},
                input_shape=[2, 16, 16, 3],
                input_dtype='float16',
                expected_output_shape=[None, 16, 16, 3],
                expected_output_dtype='float16'
            )

    def test_conv_bn_relu(self):
        norminst = Norm()
        norminst.build([None, None, None, 3])

        self.assertIsInstance(norminst.norm, layers.BatchNormalization)

    def test_policy_scope_memorize(self):
        with cnapol.policy_scope('stdconv-gn-leakyrelu'):
            norminst = Norm()
        norminst.build([None, None, None, 3])

        self.assertIsInstance(norminst.norm, norm.GroupNormalization)

        restored = Norm.from_config(norminst.get_config())
        restored.build([None, None, None, 3])

        self.assertIsInstance(restored.norm, norm.GroupNormalization)


@test_combinations.run_all_keras_modes
class TestAct(test_combinations.TestCase):
    def setUp(self):
        super(TestAct, self).setUp()
        self.default_convnormact = cnapol.global_policy()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestAct, self).tearDown()
        cnapol.set_global_policy(self.default_convnormact)
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            Act,
            kwargs={},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float32'
        )

        with cnapol.policy_scope('stdconv-gn-leakyrelu'):
            test_utils.layer_test(
                Act,
                kwargs={},
                input_shape=[2, 16, 16, 3],
                input_dtype='float32',
                expected_output_shape=[None, 16, 16, 3],
                expected_output_dtype='float32'
            )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            Act,
            kwargs={},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float16'
        )

        with cnapol.policy_scope('stdconv-gn-leakyrelu'):
            test_utils.layer_test(
                Act,
                kwargs={},
                input_shape=[2, 16, 16, 3],
                input_dtype='float16',
                expected_output_shape=[None, 16, 16, 3],
                expected_output_dtype='float16'
            )

    def test_conv_bn_relu(self):
        actinst = Act()
        actinst.build([None, None, None, 3])

        self.assertIsInstance(actinst.act, layers.ReLU)

    def test_policy_scope_memorize(self):
        with cnapol.policy_scope('stdconv-gn-leakyrelu'):
            actinst = Act()
        actinst.build([None, None, None, 3])

        self.assertIsInstance(actinst.act, layers.LeakyReLU)

        restored = Act.from_config(actinst.get_config())
        restored.build([None, None, None, 3])

        self.assertIsInstance(restored.act, layers.LeakyReLU)


@test_combinations.run_all_keras_modes
class TestConvNormAct(test_combinations.TestCase):
    def setUp(self):
        super(TestConvNormAct, self).setUp()
        self.default_convnormact = cnapol.global_policy()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestConvNormAct, self).tearDown()
        cnapol.set_global_policy(self.default_convnormact)
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            ConvNormAct,
            kwargs={'filters': 4, 'kernel_size': 3},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            ConvNormAct,
            kwargs={'filters': None, 'kernel_size': 3},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float32'
        )

        with cnapol.policy_scope('stdconv-gn-leakyrelu'):
            test_utils.layer_test(
                ConvNormAct,
                kwargs={'filters': 4, 'kernel_size': 3},
                input_shape=[2, 16, 16, 3],
                input_dtype='float32',
                expected_output_shape=[None, 16, 16, 4],
                expected_output_dtype='float32'
            )
            test_utils.layer_test(
                ConvNormAct,
                kwargs={'filters': None, 'kernel_size': 3},
                input_shape=[2, 16, 16, 3],
                input_dtype='float32',
                expected_output_shape=[None, 16, 16, 3],
                expected_output_dtype='float32'
            )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            ConvNormAct,
            kwargs={'filters': 4, 'kernel_size': 3},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 4],
            expected_output_dtype='float16'
        )
        test_utils.layer_test(
            ConvNormAct,
            kwargs={'filters': None, 'kernel_size': 3},
            input_shape=[2, 16, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float16'
        )

        with cnapol.policy_scope('stdconv-gn-leakyrelu'):
            test_utils.layer_test(
                ConvNormAct,
                kwargs={'filters': 4, 'kernel_size': 3},
                input_shape=[2, 16, 16, 3],
                input_dtype='float16',
                expected_output_shape=[None, 16, 16, 4],
                expected_output_dtype='float16'
            )
            test_utils.layer_test(
                ConvNormAct,
                kwargs={'filters': None, 'kernel_size': 3},
                input_shape=[2, 16, 16, 3],
                input_dtype='float16',
                expected_output_shape=[None, 16, 16, 3],
                expected_output_dtype='float16'
            )

    def test_conv_bn_relu(self):
        cna = ConvNormAct(4, 3)
        cna.build([None, None, None, 3])

        self.assertIsInstance(cna.conv, Conv)
        self.assertIsInstance(cna.conv.conv, sameconv.SameConv)
        self.assertEqual(cna.conv.conv.filters, 4)
        self.assertTupleEqual(cna.conv.conv.kernel_size, (3, 3))
        self.assertIsInstance(cna.norm, Norm)
        self.assertIsInstance(cna.norm.norm, layers.BatchNormalization)
        self.assertIsInstance(cna.act, Act)
        self.assertIsInstance(cna.act.act, layers.ReLU)

    def test_dwconv_bn_relu(self):
        cna = ConvNormAct(None, 3)
        cna.build([None, None, None, 3])

        self.assertIsInstance(cna.conv, Conv)
        self.assertIsInstance(cna.conv.conv, sameconv.SameDepthwiseConv)
        self.assertTupleEqual(cna.conv.conv.kernel_size, (3, 3))

    def test_policy_scope_memorize(self):
        with cnapol.policy_scope('stdconv-gn-leakyrelu'):
            cna = ConvNormAct(4, 3)
        cna.build([None, None, None, 3])

        self.assertIsInstance(cna.policy, cnapol.ConvNormActPolicy)
        self.assertEqual(cna.policy.name, 'stdconv-gn-leakyrelu')

        self.assertIsInstance(cna.conv, Conv)
        self.assertIsInstance(cna.conv.conv, sameconv.SameStandardizedConv)
        self.assertEqual(cna.conv.conv.filters, 4)
        self.assertTupleEqual(cna.conv.conv.kernel_size, (3, 3))
        self.assertIsInstance(cna.norm, Norm)
        self.assertIsInstance(cna.norm.norm, norm.GroupNormalization)
        self.assertIsInstance(cna.act, Act)
        self.assertIsInstance(cna.act.act, layers.LeakyReLU)

        restored = ConvNormAct.from_config(cna.get_config())
        restored.build([None, None, None, 3])
        self.assertIsInstance(restored.conv, Conv)
        self.assertIsInstance(restored.conv.conv, sameconv.SameStandardizedConv)
        self.assertEqual(restored.conv.conv.filters, 4)
        self.assertTupleEqual(restored.conv.conv.kernel_size, (3, 3))
        self.assertIsInstance(restored.norm, Norm)
        self.assertIsInstance(restored.norm.norm, norm.GroupNormalization)
        self.assertIsInstance(restored.act, Act)
        self.assertIsInstance(restored.act.act, layers.LeakyReLU)

    def test_policy_override_kwargs(self):
        with cnapol.policy_scope('stdconv-gn-leakyrelu'):
            cna = ConvNormAct(4, 3, strides=2)
        cna.build([None, None, None, 3])

        restored = ConvNormAct.from_config(cna.get_config())
        restored.build([None, None, None, 3])
        self.assertIsInstance(restored.conv, Conv)
        self.assertIsInstance(restored.conv.conv, sameconv.SameStandardizedConv)
        self.assertTupleEqual(restored.conv.conv.strides, (2, 2))
        self.assertIsInstance(restored.norm, Norm)
        self.assertIsInstance(restored.norm.norm, norm.GroupNormalization)
        self.assertIsInstance(restored.act, Act)
        self.assertIsInstance(restored.act.act, layers.LeakyReLU)


if __name__ == '__main__':
    tf.test.main()

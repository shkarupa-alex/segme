from keras.src import layers
from keras.src import testing

from segme.common.convnormact import Act
from segme.common.convnormact import Conv
from segme.common.convnormact import ConvAct
from segme.common.convnormact import ConvNorm
from segme.common.convnormact import ConvNormAct
from segme.common.convnormact import Norm
from segme.policy import cnapol
from segme.policy import conv
from segme.policy import norm


class TestConv(testing.TestCase):
    def setUp(self):
        super(TestConv, self).setUp()
        self.default_convnormact = cnapol.global_policy()

    def tearDown(self):
        super(TestConv, self).tearDown()
        cnapol.set_global_policy(self.default_convnormact)

    def test_layer(self):
        self.run_layer_test(
            Conv,
            init_kwargs={"filters": 4, "kernel_size": 3},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            Conv,
            init_kwargs={"filters": None, "kernel_size": 3},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 3),
            expected_output_dtype="float32",
        )

        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            self.run_layer_test(
                Conv,
                init_kwargs={"filters": 4, "kernel_size": 3},
                input_shape=(2, 16, 16, 3),
                input_dtype="float32",
                expected_output_shape=(2, 16, 16, 4),
                expected_output_dtype="float32",
            )
            self.run_layer_test(
                Conv,
                init_kwargs={"filters": None, "kernel_size": 3},
                input_shape=(2, 16, 16, 3),
                input_dtype="float32",
                expected_output_shape=(2, 16, 16, 3),
                expected_output_dtype="float32",
            )

    def test_conv_bn_relu(self):
        convinst = Conv(4, 3)
        convinst.build([None, None, None, 3])

        self.assertIsInstance(convinst, conv.FixedConv)
        self.assertEqual(convinst.filters, 4)
        self.assertTupleEqual(convinst.kernel_size, (3, 3))

    def test_dwconv_bn_relu(self):
        convinst = Conv(None, 3)
        convinst.build([None, None, None, 3])

        self.assertIsInstance(convinst, conv.FixedDepthwiseConv)
        self.assertTupleEqual(convinst.kernel_size, (3, 3))

    def test_policy_scope(self):
        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            convinst = Conv(4, 3)
        convinst.build([None, None, None, 3])

        self.assertIsInstance(convinst, conv.SpectralConv)
        self.assertEqual(convinst.filters, 4)
        self.assertTupleEqual(convinst.kernel_size, (3, 3))


class TestNorm(testing.TestCase):
    def setUp(self):
        super(TestNorm, self).setUp()
        self.default_convnormact = cnapol.global_policy()

    def tearDown(self):
        super(TestNorm, self).tearDown()
        cnapol.set_global_policy(self.default_convnormact)

    def test_layer(self):
        self.run_layer_test(
            Norm,
            init_kwargs={},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 3),
            expected_output_dtype="float32",
        )

        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            self.run_layer_test(
                Norm,
                init_kwargs={},
                input_shape=(2, 16, 16, 3),
                input_dtype="float32",
                expected_output_shape=(2, 16, 16, 3),
                expected_output_dtype="float32",
            )

    def test_conv_bn_relu(self):
        norminst = Norm()
        norminst.build([None, None, None, 3])

        self.assertIsInstance(norminst, norm.BatchNorm)

    def test_policy_scope(self):
        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            norminst = Norm()
        norminst.build([None, None, None, 3])

        self.assertIsInstance(norminst, norm.GroupNorm)


class TestAct(testing.TestCase):
    def setUp(self):
        super(TestAct, self).setUp()
        self.default_convnormact = cnapol.global_policy()

    def tearDown(self):
        super(TestAct, self).tearDown()
        cnapol.set_global_policy(self.default_convnormact)

    def test_layer(self):
        self.run_layer_test(
            Act,
            init_kwargs={},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 3),
            expected_output_dtype="float32",
        )

        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            self.run_layer_test(
                Act,
                init_kwargs={},
                input_shape=(2, 16, 16, 3),
                input_dtype="float32",
                expected_output_shape=(2, 16, 16, 3),
                expected_output_dtype="float32",
            )

    def test_conv_bn_relu(self):
        actinst = Act()
        actinst.build([None, None, None, 3])

        self.assertIsInstance(actinst, layers.ReLU)

    def test_policy_scope(self):
        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            actinst = Act()
        actinst.build([None, None, None, 3])

        self.assertIsInstance(actinst, layers.LeakyReLU)


class TestConvNormAct(testing.TestCase):
    def setUp(self):
        super(TestConvNormAct, self).setUp()
        self.default_convnormact = cnapol.global_policy()

    def tearDown(self):
        super(TestConvNormAct, self).tearDown()
        cnapol.set_global_policy(self.default_convnormact)

    def test_layer(self):
        self.run_layer_test(
            ConvNormAct,
            init_kwargs={"filters": 4, "kernel_size": 3},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            ConvNormAct,
            init_kwargs={"filters": None, "kernel_size": 3},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 3),
            expected_output_dtype="float32",
        )

        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            self.run_layer_test(
                ConvNormAct,
                init_kwargs={"filters": 4, "kernel_size": 3},
                input_shape=(2, 16, 16, 3),
                input_dtype="float32",
                expected_output_shape=(2, 16, 16, 4),
                expected_output_dtype="float32",
            )
            self.run_layer_test(
                ConvNormAct,
                init_kwargs={"filters": None, "kernel_size": 3},
                input_shape=(2, 16, 16, 3),
                input_dtype="float32",
                expected_output_shape=(2, 16, 16, 3),
                expected_output_dtype="float32",
            )

    def test_conv_bn_relu(self):
        cna = ConvNormAct(4, 3)
        cna.build([None, None, None, 3])

        self.assertIsInstance(cna.conv, conv.FixedConv)
        self.assertEqual(cna.conv.filters, 4)
        self.assertTupleEqual(cna.conv.kernel_size, (3, 3))
        self.assertIsInstance(cna.norm, norm.BatchNorm)
        self.assertIsInstance(cna.act, layers.ReLU)

    def test_dwconv_bn_relu(self):
        cna = ConvNormAct(None, 3)
        cna.build([None, None, None, 3])

        self.assertIsInstance(cna.conv, conv.FixedDepthwiseConv)
        self.assertTupleEqual(cna.conv.kernel_size, (3, 3))

    def test_policy_scope_memorize(self):
        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            cna = ConvNormAct(4, 3)
        cna.build([None, None, None, 3])

        self.assertIsInstance(cna.policy, cnapol.ConvNormActPolicy)
        self.assertEqual(cna.policy.name, "snconv-gn-leakyrelu")

        self.assertIsInstance(cna.conv, conv.SpectralConv)
        self.assertEqual(cna.conv.filters, 4)
        self.assertTupleEqual(cna.conv.kernel_size, (3, 3))
        self.assertIsInstance(cna.norm, norm.GroupNorm)
        self.assertIsInstance(cna.act, layers.LeakyReLU)

        restored = ConvNormAct.from_config(cna.get_config())
        restored.build([None, None, None, 3])
        self.assertIsInstance(restored.conv, conv.SpectralConv)
        self.assertEqual(restored.conv.filters, 4)
        self.assertTupleEqual(restored.conv.kernel_size, (3, 3))
        self.assertIsInstance(restored.norm, norm.GroupNorm)
        self.assertIsInstance(restored.act, layers.LeakyReLU)

    def test_policy_override_kwargs(self):
        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            cna = ConvNormAct(4, 3, strides=2)
        cna.build([None, None, None, 3])

        restored = ConvNormAct.from_config(cna.get_config())
        restored.build([None, None, None, 3])
        self.assertIsInstance(restored.conv, conv.SpectralConv)
        self.assertTupleEqual(restored.conv.strides, (2, 2))
        self.assertIsInstance(restored.norm, norm.GroupNorm)
        self.assertIsInstance(restored.act, layers.LeakyReLU)


class TestConvNorm(testing.TestCase):
    def setUp(self):
        super(TestConvNorm, self).setUp()
        self.default_convnormact = cnapol.global_policy()

    def tearDown(self):
        super(TestConvNorm, self).tearDown()
        cnapol.set_global_policy(self.default_convnormact)

    def test_layer(self):
        self.run_layer_test(
            ConvNorm,
            init_kwargs={"filters": 4, "kernel_size": 3},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            ConvNorm,
            init_kwargs={"filters": None, "kernel_size": 3},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 3),
            expected_output_dtype="float32",
        )

        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            self.run_layer_test(
                ConvNorm,
                init_kwargs={"filters": 4, "kernel_size": 3},
                input_shape=(2, 16, 16, 3),
                input_dtype="float32",
                expected_output_shape=(2, 16, 16, 4),
                expected_output_dtype="float32",
            )
            self.run_layer_test(
                ConvNorm,
                init_kwargs={"filters": None, "kernel_size": 3},
                input_shape=(2, 16, 16, 3),
                input_dtype="float32",
                expected_output_shape=(2, 16, 16, 3),
                expected_output_dtype="float32",
            )

    def test_conv_bn_relu(self):
        cna = ConvNorm(4, 3)
        cna.build([None, None, None, 3])

        self.assertIsInstance(cna.conv, conv.FixedConv)
        self.assertEqual(cna.conv.filters, 4)
        self.assertTupleEqual(cna.conv.kernel_size, (3, 3))
        self.assertIsInstance(cna.norm, norm.BatchNorm)

    def test_dwconv_bn_relu(self):
        cna = ConvNorm(None, 3)
        cna.build([None, None, None, 3])

        self.assertIsInstance(cna.conv, conv.FixedDepthwiseConv)
        self.assertTupleEqual(cna.conv.kernel_size, (3, 3))

    def test_policy_scope_memorize(self):
        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            cna = ConvNorm(4, 3)
        cna.build([None, None, None, 3])

        self.assertIsInstance(cna.policy, cnapol.ConvNormActPolicy)
        self.assertEqual(cna.policy.name, "snconv-gn-leakyrelu")

        self.assertIsInstance(cna.conv, conv.SpectralConv)
        self.assertEqual(cna.conv.filters, 4)
        self.assertTupleEqual(cna.conv.kernel_size, (3, 3))
        self.assertIsInstance(cna.norm, norm.GroupNorm)

        restored = ConvNorm.from_config(cna.get_config())
        restored.build([None, None, None, 3])
        self.assertIsInstance(restored.conv, conv.SpectralConv)
        self.assertEqual(restored.conv.filters, 4)
        self.assertTupleEqual(restored.conv.kernel_size, (3, 3))
        self.assertIsInstance(restored.norm, norm.GroupNorm)

    def test_policy_override_kwargs(self):
        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            cna = ConvNorm(4, 3, strides=2)
        cna.build([None, None, None, 3])

        restored = ConvNorm.from_config(cna.get_config())
        restored.build([None, None, None, 3])
        self.assertIsInstance(restored.conv, conv.SpectralConv)
        self.assertTupleEqual(restored.conv.strides, (2, 2))
        self.assertIsInstance(restored.norm, norm.GroupNorm)


class TestConvAct(testing.TestCase):
    def setUp(self):
        super(TestConvAct, self).setUp()
        self.default_ConvAct = cnapol.global_policy()

    def tearDown(self):
        super(TestConvAct, self).tearDown()
        cnapol.set_global_policy(self.default_ConvAct)

    def test_layer(self):
        self.run_layer_test(
            ConvAct,
            init_kwargs={"filters": 4, "kernel_size": 3},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            ConvAct,
            init_kwargs={"filters": None, "kernel_size": 3},
            input_shape=(2, 16, 16, 3),
            input_dtype="float32",
            expected_output_shape=(2, 16, 16, 3),
            expected_output_dtype="float32",
        )

        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            self.run_layer_test(
                ConvAct,
                init_kwargs={"filters": 4, "kernel_size": 3},
                input_shape=(2, 16, 16, 3),
                input_dtype="float32",
                expected_output_shape=(2, 16, 16, 4),
                expected_output_dtype="float32",
            )
            self.run_layer_test(
                ConvAct,
                init_kwargs={"filters": None, "kernel_size": 3},
                input_shape=(2, 16, 16, 3),
                input_dtype="float32",
                expected_output_shape=(2, 16, 16, 3),
                expected_output_dtype="float32",
            )

    def test_conv_bn_relu(self):
        cna = ConvAct(4, 3)
        cna.build([None, None, None, 3])

        self.assertIsInstance(cna.conv, conv.FixedConv)
        self.assertEqual(cna.conv.filters, 4)
        self.assertTupleEqual(cna.conv.kernel_size, (3, 3))
        self.assertIsInstance(cna.act, layers.ReLU)

    def test_dwconv_bn_relu(self):
        cna = ConvAct(None, 3)
        cna.build([None, None, None, 3])

        self.assertIsInstance(cna.conv, conv.FixedDepthwiseConv)
        self.assertTupleEqual(cna.conv.kernel_size, (3, 3))

    def test_policy_scope_memorize(self):
        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            cna = ConvAct(4, 3)
        cna.build([None, None, None, 3])

        self.assertIsInstance(cna.policy, cnapol.ConvNormActPolicy)
        self.assertEqual(cna.policy.name, "snconv-gn-leakyrelu")

        self.assertIsInstance(cna.conv, conv.SpectralConv)
        self.assertEqual(cna.conv.filters, 4)
        self.assertTupleEqual(cna.conv.kernel_size, (3, 3))
        self.assertIsInstance(cna.act, layers.LeakyReLU)

        restored = ConvAct.from_config(cna.get_config())
        restored.build([None, None, None, 3])
        self.assertIsInstance(restored.conv, conv.SpectralConv)
        self.assertEqual(restored.conv.filters, 4)
        self.assertTupleEqual(restored.conv.kernel_size, (3, 3))
        self.assertIsInstance(restored.act, layers.LeakyReLU)

    def test_policy_override_kwargs(self):
        with cnapol.policy_scope("snconv-gn-leakyrelu"):
            cna = ConvAct(4, 3, strides=2)
        cna.build([None, None, None, 3])

        restored = ConvAct.from_config(cna.get_config())
        restored.build([None, None, None, 3])
        self.assertIsInstance(restored.conv, conv.SpectralConv)
        self.assertTupleEqual(restored.conv.strides, (2, 2))
        self.assertIsInstance(restored.act, layers.LeakyReLU)

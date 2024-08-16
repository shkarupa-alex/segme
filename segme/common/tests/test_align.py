import numpy as np
import tensorflow as tf
from keras.src import testing

from segme.common.align import Align
from segme.policy import alpol
from segme.policy.align import align


class TestAlign(testing.TestCase):
    def setUp(self):
        super(TestAlign, self).setUp()
        self.default_Alignnormact = alpol.global_policy()

    def tearDown(self):
        super(TestAlign, self).tearDown()
        alpol.set_global_policy(self.default_Alignnormact)

    def test_layer(self):
        self.run_layer_test(
            Align,
            init_kwargs={"filters": 4},
            input_shape=((2, 16, 16, 3), (2, 8, 8, 6)),
            input_dtype=("float32",) * 2,
            expected_output_shape=(2, 16, 16, 4),
            expected_output_dtype="float32",
        )

        with alpol.policy_scope("deconv4"):
            self.run_layer_test(
                Align,
                init_kwargs={"filters": 4},
                input_shape=((2, 16, 16, 3), (2, 8, 8, 6)),
                input_dtype=("float32",)* 2,
                expected_output_shape=(2, 16, 16, 4),
                expected_output_dtype="float32",
            )

    def test_linear(self):
        aligninst = Align(4)
        aligninst.build([(None, None, None, 3), (None, None, None, 3)])

        self.assertIsInstance(aligninst, align.BilinearFeatureAlignment)

    def test_policy_scope(self):
        with alpol.policy_scope("sapa"):
            aligninst = Align(4)
        aligninst.build([(None, None, None, 3), (None, None, None, 3)])

        self.assertIsInstance(aligninst, align.SapaFeatureAlignment)

    def test_shape(self):
        fine = np.zeros((2, 16, 16, 16), dtype="float32")
        coarse = np.zeros((2, 8, 8, 32), dtype="float32")

        for method in align.ALIGNERS.keys():
            with alpol.policy_scope(method):
                result = Align(4)([fine, coarse])
                self.assertListEqual(result.shape.as_list(), [2, 16, 16, 4])

import numpy as np
import tensorflow as tf
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.common.grn import GRN


@test_combinations.run_all_keras_modes
class TestGRN(test_combinations.TestCase):
    def setUp(self):
        super(TestGRN, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestGRN, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            GRN,
            kwargs={'center': True, 'scale': True},
            input_shape=[2, 4, 4, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GRN,
            kwargs={'center': True, 'scale': False},
            input_shape=[2, 4, 4, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GRN,
            kwargs={'center': False, 'scale': True},
            input_shape=[2, 4, 4, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            GRN,
            kwargs={'center': False, 'scale': False},
            input_shape=[2, 4, 4, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 3],
            expected_output_dtype='float32'
        )

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            GRN,
            kwargs={'center': True, 'scale': True},
            input_shape=[2, 4, 4, 3],
            input_dtype='float16',
            expected_output_shape=[None, 4, 4, 3],
            expected_output_dtype='float16'
        )

    def test_value(self):
        inputs = np.arange(2 * 4 * 4 * 3).reshape(2, 4, 4, 3).astype('float32') / 16
        # expected = np.array([  # divisive normalization
        #     0., 0.12499277, 0.25395697, 0.36908615, 0.4999711, 0.6348924, 0.7381723, 0.8749494, 1.0158279, 1.1072583,
        #     1.2499278, 1.3967633, 1.4763446, 1.6249061, 1.7776988, 1.8454306, 1.9998844, 2.1586342, 2.2145166,
        #     2.3748627, 2.5395696, 2.583603, 2.749841, 2.920505, 2.9526892, 3.1248193, 3.3014405, 3.3217752, 3.4997976,
        #     3.682376, 3.6908612, 3.874776, 4.0633116, 4.0599475, 4.249754, 4.444247, 4.4290333, 4.6247325, 4.825182,
        #     4.7981195, 4.999711, 5.2061176, 5.167206, 5.374689, 5.5870533, 5.536292, 5.7496676, 5.9679885, 5.9595585,
        #     6.124993, 6.2921333, 6.3320312, 6.4999924, 6.6696615, 6.704503, 6.874992, 7.0471897, 7.076976, 7.2499914,
        #     7.4247174, 7.449448, 7.6249914, 7.802245, 7.8219204, 7.9999905, 8.179773, 8.194393, 8.37499, 8.5573015,
        #     8.566866, 8.74999, 8.93483, 8.939338, 9.1249895, 9.312357, 9.31181, 9.499989, 9.689886, 9.684282,
        #     9.874989, 10.067413, 10.056755, 10.249989, 10.4449415, 10.429228, 10.624988, 10.82247, 10.8017, 10.999987,
        #     11.199997, 11.174172, 11.374987, 11.577526, 11.546644, 11.749987, 11.955053],
        #     'float32').reshape(2, 4, 4, 3)
        expected = np.array([  # standardization
            0.0, 0.0622203, 0.2783629, -0.0417066, 0.2488811, 0.6959072, -0.0834133, 0.435542, 1.1134515, -0.12512,
            0.6222028, 1.5309958, -0.1668266, 0.8088636, 1.94854, -0.2085333, 0.9955245, 2.3660843, -0.2502399,
            1.1821853, 2.7836287, -0.2919466, 1.3688462, 3.201173, -0.3336532, 1.555507, 3.6187172, -0.3753599,
            1.7421678, 4.0362616, -0.4170665, 1.9288286, 4.453806, -0.4587732, 2.1154895, 4.8713503, -0.5004798,
            2.3021502, 5.2888947, -0.5421865, 2.4888113, 5.706439, -0.5838931, 2.675472, 6.1239834, -0.6255998,
            2.8621328, 6.5415273, -0.6737226, 3.061888, 6.9524794, -0.7158302, 3.2493505, 7.369627, -0.757938, 3.436813,
            7.786776, -0.8000456, 3.6242757, 8.203925, -0.8421533, 3.811738, 8.621074, -0.884261, 3.9992003, 9.038222,
            -0.9263686, 4.186663, 9.455371, -0.9684763, 4.3741255, 9.8725195, -1.0105839, 4.5615883, 10.289668,
            -1.0526916, 4.7490506, 10.706818, -1.0947993, 4.9365134, 11.123966, -1.136907, 5.1239758, 11.541115,
            -1.1790146, 5.311438, 11.958263, -1.2211223, 5.498901, 12.375412, -1.26323, 5.686363, 12.792561, -1.3053375,
            5.873826, 13.209709], 'float32').reshape(2, 4, 4, 3)
        layer = GRN(epsilon=1.001e-5)

        result = layer(inputs)
        result = self.evaluate(result)
        self.assertAllClose(expected, result, atol=7e-4)


if __name__ == '__main__':
    tf.test.main()
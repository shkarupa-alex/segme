import numpy as np
import tensorflow as tf
from tf_keras import mixed_precision
from tf_keras.src.testing_infra import test_combinations, test_utils
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
        expected = np.array([
            0.0, 0.1249438, 0.2579502, 0.3632413, 0.4997752, 0.6448756, 0.7264826, 0.8746065, 1.031801, 1.0897238,
            1.2494379, 1.4187263, 1.4529651, 1.6242692, 1.8056517, 1.8162065, 1.9991007, 2.1925771, 2.1794477,
            2.3739321, 2.5795026, 2.5426891, 2.7487636, 2.9664278, 2.9059303, 3.1235948, 3.3533533, 3.2691715,
            3.4984262, 3.7402785, 3.6324129, 3.8732576, 4.1272039, 3.9956541, 4.2480888, 4.5141292, 4.3588953,
            4.6229205, 4.9010549, 4.7221365, 4.9977517, 5.2879801, 5.0853782, 5.3725829, 5.6749053, 5.4486194,
            5.7474146, 6.061831, 5.9193077, 6.1246138, 6.334445, 6.2892647, 6.4995899, 6.7145114, 6.6592212, 6.8745666,
            7.0945783, 7.0291777, 7.2495427, 7.4746451, 7.3991346, 7.6245193, 7.8547115, 7.7690916, 7.9994955,
            8.2347784, 8.1390476, 8.3744717, 8.6148453, 8.5090046, 8.7494478, 8.9949121, 8.8789616, 9.1244249,
            9.3749781, 9.2489185, 9.4994011, 9.7550449, 9.6188755, 9.8743773, 10.1351118, 9.9888315, 10.2493534,
            10.5151787, 10.3587885, 10.6243296, 10.8952456, 10.7287455, 10.9993067, 11.2753115, 11.0987015, 11.3742828,
            11.6553783, 11.4686584, 11.749259, 12.0354452],
            'float32').reshape(2, 4, 4, 3)
        layer = GRN(epsilon=1.001e-5)

        result = layer(inputs)
        result = self.evaluate(result)
        self.assertAllClose(expected, result, atol=7e-4)


if __name__ == '__main__':
    tf.test.main()

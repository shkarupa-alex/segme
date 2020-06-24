# import tensorflow as tf
# from tensorflow.python.keras import keras_parameterized, testing_utils
# from ..upconv import DexiNedUpConvBlock
#
#
# @keras_parameterized.run_all_keras_modes
# class TestDexiNedUpConvBlock(keras_parameterized.TestCase):
#     def test_layer(self):
#         testing_utils.layer_test(
#             DexiNedUpConvBlock,
#             kwargs={'up_scale': 2},
#             input_shape=[2, 16, 16, 3],
#             input_dtype='float32',
#             expected_output_shape=[None, 64, 64, 1],
#             expected_output_dtype='float32'
#         )
#
#         const_init = tf.keras.initializers.constant(0.1)
#         testing_utils.layer_test(
#             DexiNedUpConvBlock,
#             kwargs={'up_scale': 3, 'kernel_initializer': const_init,
#                     'kernel_l2': 0.01, 'bias_initializer': const_init},
#             input_shape=[2, 16, 16, 3],
#             input_dtype='float32',
#             expected_output_shape=[None, 128, 128, 1],
#             expected_output_dtype='float32'
#         )
#
#
# if __name__ == '__main__':
#     tf.test.main()

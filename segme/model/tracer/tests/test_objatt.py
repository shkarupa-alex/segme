# import tensorflow as tf
# from keras import keras_parameterized
# from ..objatt import ObjectAttention
# from ....testing_utils import layer_multi_io_test
#
#
# @keras_parameterized.run_all_keras_modes
# class TestObjectAttention(keras_parameterized.TestCase):
#     def test_layer(self):
#         layer_multi_io_test(
#             ObjectAttention,
#             kwargs={'kernel_size': 3, 'denoise': 0.93},
#             input_shapes=[(2, 32, 32, 16), (2, 32, 32, 1)],
#             input_dtypes=['float32'] * 2,
#             expected_output_shapes=[(None, 32, 32, 1)],
#             expected_output_dtypes=['float32']
#         )
#
#
# if __name__ == '__main__':
#     tf.test.main()

# import numpy as np
# import tensorflow as tf
# from tensorflow.python.keras import keras_parameterized
# from ..testing_utils import layer_multi_io_test
#
#
# class OneToManyLayer(tf.keras.layers.Dense):
#     def call(self, inputs):
#         result = super(OneToManyLayer, self).call(inputs)
#
#         return result, result + result
#
#     def compute_output_shape(self, input_shape):
#         result = super(OneToManyLayer, self).compute_output_shape(input_shape)
#
#         return result, result
#
#
# @keras_parameterized.run_all_keras_modes
# class LayerMultiIOTestTest(keras_parameterized.TestCase):
#     def test_one_to_one(self):
#         layer_multi_io_test(
#             tf.keras.layers.Dense,
#             kwargs={'units': 10},
#             input_shapes=[(2, 4)]
#         )
#         layer_multi_io_test(
#             tf.keras.layers.Dense,
#             kwargs={'units': 10},
#             input_datas=[np.random.random((2, 4)).astype(np.float32)]
#         )
#         layer_multi_io_test(
#             tf.keras.layers.Dense,
#             kwargs={'units': 10},
#             input_shapes=[(2, 4)],
#             expected_output_dtypes=['float32']
#         )
#
#         self.assertEqual(layer_multi_io_test(
#             tf.keras.layers.Dense,
#             kwargs={'units': 10, 'dtype': 'float16'},
#             input_shapes=[(2, 4)],
#             input_dtypes=['float16']
#         ).dtype.name, 'float16')
#         self.assertEqual(layer_multi_io_test(
#             tf.keras.layers.Dense,
#             kwargs={'units': 10, 'dtype': 'float16'},
#             input_datas=[np.random.random((2, 4))],
#             input_dtypes=['float16']
#         ).dtype.name, 'float16')
#         self.assertEqual(layer_multi_io_test(
#             tf.keras.layers.Dense,
#             kwargs={'units': 10, 'dtype': 'float16'},
#             input_shapes=[(2, 4)],
#             input_dtypes=['float16'],
#             expected_output_dtypes=['float16']
#         ).dtype.name, 'float16')
#
#     def test_many_to_one(self):
#         layer_multi_io_test(
#             tf.keras.layers.Add,
#             input_shapes=[(2, 4), (2, 4)],
#         )
#
#     def test_one_to_many(self):
#         with tf.keras.utils.custom_object_scope(
#                 {'OneToManyLayer': OneToManyLayer}):
#             layer_multi_io_test(
#                 OneToManyLayer,
#                 kwargs={'units': 10},
#                 input_shapes=[(2, 4)],
#                 expected_output_dtypes=['float32', 'float32']
#             )
#
#
# if __name__ == "__main__":
#     tf.test.main()

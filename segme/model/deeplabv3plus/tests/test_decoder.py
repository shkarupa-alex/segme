# import numpy as np
# import tensorflow as tf
# from tensorflow.python.keras import keras_parameterized, testing_utils
# from ..decoder import DeepLabV3PlusDecoder
#
#
# @keras_parameterized.run_all_keras_modes
# class TestDeepLabV3PlusDecoder(keras_parameterized.TestCase):
#     def test_layer(self):
#         layer = DeepLabV3PlusDecoder(8, 4)
#
#         input_shape0 = [2, 64, 64, 10]
#         input_dtype0 = 'float32'
#         input_data0 = 10 * np.random.random(input_shape0) - 0.5
#         input_data0 = input_data0.astype(input_dtype0)
#
#         input_shape1 = [2, 16, 16, 3]
#         input_dtype1 = 'float32'
#         input_data1 = 10 * np.random.random(input_shape1) - 0.5
#         input_data1 = input_data1.astype(input_dtype1)
#
#         input_shape2 = [2, 4, 4, 3]
#         input_dtype2 = 'float32'
#         input_data2 = 10 * np.random.random(input_shape2) - 0.5
#         input_data2 = input_data2.astype(input_dtype2)
#
#         expected_output_shape = [None, 64, 64, 4]
#         expected_output_dtype = 'float32'
#
#         # test get_weights , set_weights at layer level
#         weights = layer.get_weights()
#         layer.set_weights(weights)
#
#         # test in functional API
#         x0 = tf.keras.layers.Input(shape=input_shape0[1:], dtype=input_dtype0)
#         x1 = tf.keras.layers.Input(shape=input_shape1[1:], dtype=input_dtype1)
#         x2 = tf.keras.layers.Input(shape=input_shape2[1:], dtype=input_dtype2)
#         y = layer([x0, x1, x2])
#         self.assertEqual(tf.keras.backend.dtype(y), expected_output_dtype)
#         self.assertEqual(y.get_shape().as_list(), expected_output_shape)
#
#         # check shape inference
#         model = tf.keras.models.Model([x0, x1, x2], y)
#         computed_output_shape = tuple(layer.compute_output_shape([
#             input_shape0, input_shape1, input_shape2]).as_list())
#         computed_output_signature = layer.compute_output_signature([
#             tf.TensorSpec(shape=input_shape0, dtype=input_dtype0),
#             tf.TensorSpec(shape=input_shape1, dtype=input_dtype1),
#             tf.TensorSpec(shape=input_shape2, dtype=input_dtype2),
#         ])
#         actual_output = model.predict([input_data0, input_data1, input_data2])
#         actual_output_shape = actual_output.shape
#
#         self.assertEqual(computed_output_shape, actual_output_shape)
#         self.assertEqual(computed_output_signature.shape, actual_output_shape)
#         self.assertEqual(computed_output_signature.dtype, actual_output.dtype)
#
#         # test serialization, weight setting at model level
#         model_config = model.get_config()
#         recovered_model = tf.keras.models.Model.from_config(model_config)
#         if model.weights:
#             weights = model.get_weights()
#             recovered_model.set_weights(weights)
#             output = recovered_model.predict([
#                 input_data0, input_data1, input_data2])
#             self.assertAllClose(output, actual_output, rtol=1e-3, atol=1e-6)
#
#         # test training mode (e.g. useful for dropout tests)
#         # Rebuild the model to avoid the graph being reused between predict()
#         # See b/120160788 for more details. This should be mitigated after 2.0.
#         model = tf.keras.models.Model([x0, x1, x2], layer([x0, x1, x2]))
#         model.compile(
#             'rmsprop',
#             'mse',
#             weighted_metrics=['acc'],
#             run_eagerly=testing_utils.should_run_eagerly())
#         model.train_on_batch([
#             input_data0, input_data1, input_data2], actual_output)
#
#
# if __name__ == '__main__':
#     tf.test.main()

# import numpy as np
# import tensorflow as tf
# from tensorflow.python.keras import keras_parameterized, testing_utils
# from ..upsample import UpBySample2D
#
#
# @keras_parameterized.run_all_keras_modes
# class TestUpBySample2D(keras_parameterized.TestCase):
#     def test_layer(self):
#         layer = UpBySample2D()
#
#         input_shape0 = [2, 16, 16, 10]
#         input_dtype0 = 'float32'
#         input_data0 = 10 * np.random.random(input_shape0) - 0.5
#         input_data0 = input_data0.astype(input_dtype0)
#
#         input_shape1 = [2, 24, 32, 3]
#         input_dtype1 = 'float32'
#         input_data1 = 10 * np.random.random(input_shape1) - 0.5
#         input_data1 = input_data1.astype(input_dtype1)
#
#         expected_output_shape = [None, 24, 32, 10]
#         expected_output_dtype = 'float32'
#
#         # test get_weights , set_weights at layer level
#         weights = layer.get_weights()
#         layer.set_weights(weights)
#
#         # test in functional API
#         x0 = tf.keras.layers.Input(shape=input_shape0[1:], dtype=input_dtype0)
#         x1 = tf.keras.layers.Input(shape=input_shape1[1:], dtype=input_dtype1)
#         y = layer([x0, x1])
#         self.assertEqual(tf.keras.backend.dtype(y), expected_output_dtype)
#         self.assertEqual(y.get_shape().as_list(), expected_output_shape)
#
#         # check shape inference
#         model = tf.keras.models.Model([x0, x1], y)
#         computed_output_shape = tuple(
#             layer.compute_output_shape([input_shape0, input_shape1]).as_list())
#         computed_output_signature = layer.compute_output_signature([
#             tf.TensorSpec(shape=input_shape0, dtype=input_dtype0),
#             tf.TensorSpec(shape=input_shape1, dtype=input_dtype1)
#         ])
#         actual_output = model.predict([input_data0, input_data1])
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
#             output = recovered_model.predict([input_data0, input_data1])
#             self.assertAllClose(output, actual_output, rtol=1e-3, atol=1e-6)
#
#         # test training mode (e.g. useful for dropout tests)
#         # Rebuild the model to avoid the graph being reused between predict()
#         # See b/120160788 for more details. This should be mitigated after 2.0.
#         model = tf.keras.models.Model([x0, x1], layer([x0, x1]))
#         model.compile(
#             'rmsprop',
#             'mse',
#             weighted_metrics=['acc'],
#             run_eagerly=testing_utils.should_run_eagerly())
#         model.train_on_batch([input_data0, input_data1], actual_output)
#
#     def test_corners(self):
#         target = tf.reshape(tf.range(9, dtype=tf.float32), [1, 3, 3, 1])
#         sample = tf.zeros([1, 10, 9, 1], dtype=tf.float32)
#         result = UpBySample2D()([target, sample])
#         result = self.evaluate(result)
#
#         # See https://github.com/tensorflow/tensorflow/
#         # issues/6720#issuecomment-644111750
#         expected = np.array([
#             [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.],
#             [0.667, 0.917, 1.167, 1.417, 1.667, 1.917, 2.167, 2.417, 2.667],
#             [1.333, 1.583, 1.833, 2.083, 2.333, 2.583, 2.833, 3.083, 3.333],
#             [2., 2.25, 2.5, 2.75, 3., 3.25, 3.5, 3.75, 4.],
#             [2.667, 2.917, 3.167, 3.417, 3.667, 3.917, 4.167, 4.417, 4.667],
#             [3.333, 3.583, 3.833, 4.083, 4.333, 4.583, 4.833, 5.083, 5.333],
#             [4., 4.25, 4.5, 4.75, 5., 5.25, 5.5, 5.75, 6.],
#             [4.667, 4.917, 5.167, 5.417, 5.667, 5.917, 6.167, 6.417, 6.667],
#             [5.333, 5.583, 5.833, 6.083, 6.333, 6.583, 6.833, 7.083, 7.333],
#             [6., 6.25, 6.5, 6.75, 7., 7.25, 7.5, 7.75, 8.]
#         ]).reshape([1, 10, 9, 1])
#
#         self.assertAllClose(result, expected, 0.002)
#
#
# if __name__ == '__main__':
#     tf.test.main()

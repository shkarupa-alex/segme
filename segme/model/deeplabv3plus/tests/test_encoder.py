# import numpy as np
# import tensorflow as tf
# from tensorflow.python.keras import keras_parameterized, testing_utils
# from tensorflow.python.util import tf_inspect
# from ..encoder import DeepLabV3PlusEncoder
#
#
# @keras_parameterized.run_all_keras_modes
# class TestDeepLabV3PlusEncoder(keras_parameterized.TestCase):
#     def test_layer(self):
#         self._encoder_layer_test(
#             DeepLabV3PlusEncoder,
#             kwargs={
#                 'bone_arch': 'resnet50', 'bone_init': 'imagenet',
#                 'bone_train': False, 'aspp_filters': 10, 'aspp_stride': 16},
#             input_shape=[2, 224, 224, 3],
#             input_dtype='uint8',
#             expected_output_shapes=[
#                 [None, 56, 56, 256],
#                 [None, 14, 14, 10]
#             ],
#             expected_output_dtype='float32'
#         )
#
#     def _encoder_layer_test(self, layer_cls, kwargs, input_shape, input_dtype,
#                             expected_output_shapes, expected_output_dtype):
#         input_data = 10 * np.random.random(input_shape) - 0.5
#         input_data = input_data.astype(input_dtype)
#
#         # instantiation
#         kwargs = kwargs or {}
#         layer = layer_cls(**kwargs)
#
#         # test get_weights , set_weights at layer level
#         weights = layer.get_weights()
#         layer.set_weights(weights)
#
#         # test and instantiation from weights
#         if 'weights' in tf_inspect.getargspec(layer_cls.__init__):
#             kwargs['weights'] = weights
#             layer = layer_cls(**kwargs)
#
#         # test in functional API
#         x = tf.keras.layers.Input(shape=input_shape[1:], dtype=input_dtype)
#         y = layer(x)
#
#         output_size = len(y)
#         if output_size == len(expected_output_shapes) - 1:
#             expected_output_shapes = expected_output_shapes[1:]
#
#         for i in range(output_size):
#             self.assertEqual(
#                 tf.keras.backend.dtype(y[i]), expected_output_dtype)
#             self._encoder_shape_test(y[i].shape, expected_output_shapes[i])
#
#         # check shape inference
#         model = tf.keras.models.Model(x, y)
#         computed_output_shape = layer.compute_output_shape(
#             tf.TensorShape(input_shape))
#         computed_output_signature = layer.compute_output_signature(
#             tf.TensorSpec(shape=input_shape, dtype=input_dtype))
#         actual_output = model.predict(input_data)
#         actual_output_shape = [ao.shape for ao in actual_output]
#
#         for i in range(output_size):
#             self._encoder_shape_test(computed_output_shape[i],
#                                      actual_output_shape[i])
#             self._encoder_shape_test(computed_output_signature[i].shape,
#                                      actual_output_shape[i])
#             self.assertEqual(
#                 tf.keras.backend.dtype(computed_output_signature[i]),
#                 expected_output_dtype)
#
#         # test serialization, weight setting at model level
#         model_config = model.get_config()
#         recovered_model = tf.keras.models.Model.from_config(model_config)
#         if model.weights:
#             weights = model.get_weights()
#             recovered_model.set_weights(weights)
#             output = recovered_model.predict(input_data)
#
#             for i in range(output_size):
#                 self.assertAllClose(
#                     output[i], actual_output[i], rtol=1e-3, atol=1e-6)
#
#         # test training mode (e.g. useful for dropout tests)
#         # Rebuild the model to avoid the graph being reused between predict()
#         # See b/120160788 for more details. This should be mitigated after 2.0.
#         model = tf.keras.models.Model(x, layer(x))
#         model.compile(
#             'rmsprop',
#             'mse',
#             weighted_metrics=['acc'],
#             run_eagerly=testing_utils.should_run_eagerly())
#         model.train_on_batch(input_data, actual_output)
#
#         # test as first layer in Sequential API
#         layer_config = layer.get_config()
#         layer_config['batch_input_shape'] = input_shape
#         layer = layer.__class__.from_config(layer_config)
#
#         # test serialization, weight setting at model level
#         actual_output = model.predict(input_data)
#         model_config = model.get_config()
#         recovered_model = tf.keras.Model.from_config(model_config)
#         if model.weights:
#             weights = model.get_weights()
#             recovered_model.set_weights(weights)
#             output = recovered_model.predict(input_data)
#
#             for i in range(output_size):
#                 self.assertAllClose(
#                     output[i], actual_output[i], rtol=1e-3, atol=1e-6)
#
#     def _encoder_shape_test(self, tensor_shape, expected_shape):
#         actual_shape = tensor_shape.as_list()
#         expected_shape = list(expected_shape)
#
#         if expected_shape[-1] is False:
#             self.assertEqual(actual_shape[:-1] + [False], expected_shape)
#         else:
#             self.assertEqual(actual_shape, expected_shape)
#
#
# if __name__ == '__main__':
#     tf.test.main()

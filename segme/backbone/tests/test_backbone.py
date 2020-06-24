# import numpy as np
# import tensorflow as tf
# from absl.testing import parameterized
# from tensorflow.python.keras import keras_parameterized, testing_utils
# from tensorflow.python.util import tf_inspect
# from ..backbone import Backbone
#
# _CUSTOM_TESTS = {
#     'inception_v3', 'inception_resnet_v2', 'xception', 'vgg_16', 'vgg_19'}
# _DEFAULT_TEST = set(Backbone._config.keys()) - _CUSTOM_TESTS
#
# @keras_parameterized.run_all_keras_modes
# class TestBackbone(keras_parameterized.TestCase):
#     def test_arch_config(self):
#         archs = Backbone._config
#         for arch_name in archs:
#             arch = archs[arch_name]
#             self.assertIsInstance(arch, tuple)
#             self.assertLen(arch, 2)
#
#             model, feats = arch
#             self.assertTrue(callable(model))
#             self.assertIsInstance(feats, tuple)
#             self.assertLen(feats, 6)
#
#             for ft in feats:
#                 self.assertIsInstance(ft, (type(None), str, int))
#
#     @parameterized.parameters(_DEFAULT_TEST)
#     def test_layer_default_trainable(self, arch_name):
#         self._backbone_layer_test(
#             Backbone,
#             kwargs={'arch': arch_name, 'init': None, 'trainable': True},
#             input_shape=[2, 224, 224, 3],
#             input_dtype='uint8',
#             expected_output_shapes=[
#                 [None, 112, 112, False],
#                 [None, 56, 56, False],
#                 [None, 28, 28, False],
#                 [None, 14, 14, False],
#                 [None, 7, 7, False]
#             ],
#             expected_output_dtype='float32'
#         )
#
#     @parameterized.parameters(_DEFAULT_TEST)
#     def test_layer_default_imagenet(self, arch_name):
#         self._backbone_layer_test(
#             Backbone,
#             kwargs={'arch': arch_name, 'init': 'imagenet', 'trainable': False},
#             input_shape=[2, 224, 224, 3],
#             input_dtype='uint8',
#             expected_output_shapes=[
#                 [None, 112, 112, False],
#                 [None, 56, 56, False],
#                 [None, 28, 28, False],
#                 [None, 14, 14, False],
#                 [None, 7, 7, False]
#             ],
#             expected_output_dtype='float32'
#         )
#
#     # def test_layer_inception_v3_trainable(self):
#     #     self._backbone_layer_test(
#     #         Backbone,
#     #         kwargs={'arch': 'inception_v3', 'init': None, 'trainable': True},
#     #         input_shape=[2, 224, 224, 3],
#     #         input_dtype='uint8',
#     #         expected_output_shapes=[
#     #             [None, 109, 109, False],
#     #             [None, 52, 52, False],
#     #             [None, 25, 25, False],
#     #             [None, 12, 12, False],
#     #             [None, 5, 5, False]
#     #         ],
#     #         expected_output_dtype='float32'
#     #     )
#     #
#     # def test_layer_inception_v3_imagenet(self):
#     #     self._backbone_layer_test(
#     #         Backbone,
#     #         kwargs={'arch': 'inception_v3', 'init': 'imagenet',
#     #                 'trainable': False},
#     #         input_shape=[2, 224, 224, 3],
#     #         input_dtype='uint8',
#     #         expected_output_shapes=[
#     #             [None, 109, 109, False],
#     #             [None, 52, 52, False],
#     #             [None, 25, 25, False],
#     #             [None, 12, 12, False],
#     #             [None, 5, 5, False]
#     #         ],
#     #         expected_output_dtype='float32'
#     #     )
#     #
#     # def test_layer_inception_resnet_v2_trainable(self):
#     #     self._backbone_layer_test(
#     #         Backbone,
#     #         kwargs={'arch': 'inception_resnet_v2', 'init': None,
#     #                 'trainable': True},
#     #         input_shape=[2, 224, 224, 3],
#     #         input_dtype='uint8',
#     #         expected_output_shapes=[
#     #             [None, 109, 109, False],
#     #             [None, 52, 52, False],
#     #             [None, False, False, False],
#     #             [None, False, False, False],
#     #             [None, False, False, False]
#     #         ],
#     #         expected_output_dtype='float32'
#     #     )
#     #
#     # def test_layer_inception_resnet_v2_imagenet(self):
#     #     self._backbone_layer_test(
#     #         Backbone,
#     #         kwargs={'arch': 'inception_resnet_v2', 'init': 'imagenet',
#     #                 'trainable': False},
#     #         input_shape=[2, 224, 224, 3],
#     #         input_dtype='uint8',
#     #         expected_output_shapes=[
#     #             [None, 109, 109, False],
#     #             [None, 52, 52, False],
#     #             [None, False, False, False],
#     #             [None, False, False, False],
#     #             [None, False, False, False]
#     #         ],
#     #         expected_output_dtype='float32'
#     #     )
#     #
#     # def test_layer_xception_trainable(self):
#     #     self._backbone_layer_test(
#     #         Backbone,
#     #         kwargs={'arch': 'xception', 'init': None, 'trainable': True},
#     #         input_shape=[2, 224, 224, 3],
#     #         input_dtype='uint8',
#     #         expected_output_shapes=[
#     #             [None, 109, 109, False],
#     #             [None, 55, 55, False],
#     #             [None, 28, 28, False],
#     #             [None, 14, 14, False],
#     #             [None, 7, 7, False]
#     #         ],
#     #         expected_output_dtype='float32'
#     #     )
#     #
#     # def test_layer_xception_imagenet(self):
#     #     self._backbone_layer_test(
#     #         Backbone,
#     #         kwargs={'arch': 'xception', 'init': 'imagenet',
#     #                 'trainable': False},
#     #         input_shape=[2, 224, 224, 3],
#     #         input_dtype='uint8',
#     #         expected_output_shapes=[
#     #             [None, 109, 109, False],
#     #             [None, 55, 55, False],
#     #             [None, 28, 28, False],
#     #             [None, 14, 14, False],
#     #             [None, 7, 7, False]
#     #         ],
#     #         expected_output_dtype='float32'
#     #     )
#     #
#     # @parameterized.parameters(['vgg_16', 'vgg_19'])
#     # def test_layer_vgg_trainable(self, vgg_arch):
#     #     self._backbone_layer_test(
#     #         Backbone,
#     #         kwargs={'arch': vgg_arch, 'init': None, 'trainable': True},
#     #         input_shape=[2, 224, 224, 3],
#     #         input_dtype='uint8',
#     #         expected_output_shapes=[
#     #             [None, 224, 224, False],
#     #             [None, 112, 112, False],
#     #             [None, 56, 56, False],
#     #             [None, 28, 28, False],
#     #             [None, 14, 14, False]
#     #         ],
#     #         expected_output_dtype='float32'
#     #     )
#     #
#     # @parameterized.parameters(['vgg_16', 'vgg_19'])
#     # def test_layer_vgg_imagenet(self, vgg_arch):
#     #     self._backbone_layer_test(
#     #         Backbone,
#     #         kwargs={'arch': vgg_arch, 'init': 'imagenet',
#     #                 'trainable': False},
#     #         input_shape=[2, 224, 224, 3],
#     #         input_dtype='uint8',
#     #         expected_output_shapes=[
#     #             [None, 224, 224, False],
#     #             [None, 112, 112, False],
#     #             [None, 56, 56, False],
#     #             [None, 28, 28, False],
#     #             [None, 14, 14, False]
#     #         ],
#     #         expected_output_dtype='float32'
#     #     )
#
#     def _backbone_layer_test(self, layer_cls, kwargs, input_shape, input_dtype,
#                              expected_output_shapes, expected_output_dtype):
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
#         for i in range(output_size):
#             self.assertEqual(
#                 tf.keras.backend.dtype(y[i]), expected_output_dtype)
#             self._backbone_shape_test(y[i].shape, expected_output_shapes[i])
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
#             self._backbone_shape_test(tf.TensorShape(actual_output_shape[i]),
#                                       expected_output_shapes[i])
#             self._backbone_shape_test(computed_output_shape[i],
#                                       expected_output_shapes[i])
#             self._backbone_shape_test(computed_output_signature[i].shape,
#                                       expected_output_shapes[i])
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
#         # Rebuild the model to avoid the graph being reused between predict() and
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
#         recovered_model._run_eagerly = testing_utils.should_run_eagerly()
#         if model.weights:
#             weights = model.get_weights()
#             recovered_model.set_weights(weights)
#             output = recovered_model.predict(input_data)
#
#             for i in range(output_size):
#                 self.assertAllClose(
#                     output[i], actual_output[i], rtol=1e-3, atol=1e-6)
#
#     def _backbone_shape_test(self, tensor_shape, expected_shape):
#         actual_shape = tensor_shape.as_list()
#         expected_shape = list(expected_shape)
#
#         _expected_shape = []
#         _actual_shape = []
#         for ex, ac in zip(expected_shape, actual_shape):
#             if ex is False:
#                 _expected_shape.append(False)
#                 _actual_shape.append(False)
#             else:
#                 _expected_shape.append(ex)
#                 _actual_shape.append(ac)
#         if _expected_shape[0] is None:
#             _actual_shape[0] = None
#
#         self.assertEqual(_actual_shape, _expected_shape)
#
#
# if __name__ == '__main__':
#     tf.test.main()

# import numpy as np
# import tensorflow as tf
# from tensorflow.python.keras import keras_parameterized, testing_utils
# from tensorflow.python.training.tracking import util as trackable_util
# from tensorflow.python.util import object_identity
# from ..model import DeepLabV3Plus
#
#
# @keras_parameterized.run_all_keras_modes
# class TestDeepLabV3Plus(keras_parameterized.TestCase):
#     def test_layer(self):
#         testing_utils.layer_test(
#             DeepLabV3Plus,
#             kwargs={
#                 'classes': 3, 'bone_arch': 'resnet_50', 'bone_init': 'imagenet',
#                 'bone_freeze': True, 'aspp_filters': 8, 'aspp_stride': 8,
#                 'low_filters': 16, 'decoder_filters': 4
#             },
#             input_shape=[2, 224, 224, 3],
#             input_dtype='uint8',
#             expected_output_shape=[None, 224, 224, 3],
#             expected_output_dtype='float32'
#         )
#         testing_utils.layer_test(
#             DeepLabV3Plus,
#             kwargs={
#                 'classes': 2, 'bone_arch': 'resnet_50', 'bone_init': 'imagenet',
#                 'bone_freeze': True, 'aspp_filters': 8, 'aspp_stride': 16,
#                 'low_filters': 16, 'decoder_filters': 4
#             },
#             input_shape=[2, 224, 224, 3],
#             input_dtype='uint8',
#             expected_output_shape=[None, 224, 224, 1],
#             expected_output_dtype='float32'
#         )
#         testing_utils.layer_test(
#             DeepLabV3Plus,
#             kwargs={
#                 'classes': 1, 'bone_arch': 'resnet_50', 'bone_init': 'imagenet',
#                 'bone_freeze': True, 'aspp_filters': 8, 'aspp_stride': 32,
#                 'low_filters': 16, 'decoder_filters': 4
#             },
#             input_shape=[2, 224, 224, 3],
#             input_dtype='uint8',
#             expected_output_shape=[None, 224, 224, 1],
#             expected_output_dtype='float32'
#         )
#
#     def test_model(self):
#         num_classes = 5
#
#         model = tf.keras.models.Sequential()
#         model.add(DeepLabV3Plus(
#             classes=num_classes,
#             bone_arch='resnet_50',
#             bone_init='imagenet',
#             bone_freeze=True,
#             aspp_filters=8,
#             aspp_stride=16,
#             low_filters=16,
#             decoder_filters=4
#         ))
#
#         model.compile(
#             optimizer='sgd', loss='sparse_categorical_crossentropy',
#             run_eagerly=testing_utils.should_run_eagerly(),
#             experimental_run_tf_function=testing_utils.should_run_tf_function())
#         model.fit(
#             np.random.random((2, 224, 224, 3)).astype(np.uint8),
#             np.random.randint(0, num_classes, (2, 224, 224)),
#             epochs=1, batch_size=10)
#
#         # test config
#         model.get_config()
#
#         # check whether the model variables are present
#         # in the trackable list of objects
#         checkpointed_objects = object_identity.ObjectIdentitySet(
#             trackable_util.list_objects(model))
#         for v in model.variables:
#             self.assertIn(v, checkpointed_objects)
#
#
# if __name__ == '__main__':
#     tf.test.main()

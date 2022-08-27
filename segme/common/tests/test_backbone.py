# TODO
# import tensorflow as tf
# from keras.mixed_precision import policy as mixed_precision
# from keras.testing_infra import test_combinations
# from segme.common.backbone import Backbone
# from segme.policy import bbpol
# from segme.testing_utils import layer_multi_io_test
#
#
# @test_combinations.run_all_keras_modes
# class TestBackbone(test_combinations.TestCase):
#     def setUp(self):
#         super(TestBackbone, self).setUp()
#         self.default_bb = bbpol.global_policy()
#         self.default_policy = mixed_precision.global_policy()
#
#     def tearDown(self):
#         super(TestBackbone, self).tearDown()
#         bbpol.set_global_policy(self.default_bb)
#         mixed_precision.set_global_policy(self.default_policy)
#
#     def test_layer(self):
#         layer_multi_io_test(
#             Backbone,
#             kwargs={'scales': None},
#             input_shapes=[(2, 224, 224, 3)],
#             input_dtypes=['float32'],
#             expected_output_shapes=[
#                 (None, 112, 112, 64), (None, 56, 56, 256), (None, 28, 28, 512), (None, 14, 14, 1024),
#                 (None, 7, 7, 2048)],
#             expected_output_dtypes=['float32'] * 5
#         )
#         layer_multi_io_test(
#             Backbone,
#             kwargs={'scales': [2, 8]},
#             input_shapes=[(2, 224, 224, 3)],
#             input_dtypes=['uint8'],
#             expected_output_shapes=[(None, 112, 112, 64), (None, 28, 28, 512)],
#             expected_output_dtypes=['float32'] * 2
#         )
#
#         with bbpol.policy_scope('swintiny224_none'):
#             layer_multi_io_test(
#                 Backbone,
#                 kwargs={'scales': None},
#                 input_shapes=[(2, 224, 224, 3)],
#                 input_dtypes=['float32'],
#                 expected_output_shapes=[
#                     (None, 56, 56, 96), (None, 28, 28, 192), (None, 14, 14, 384), (None, 7, 7, 768)],
#                 expected_output_dtypes=['float32'] * 4
#             )
#             layer_multi_io_test(
#                 Backbone,
#                 kwargs={'scales': None},
#                 input_shapes=[(2, 224, 224, 3)],
#                 input_dtypes=['uint8'],
#                 expected_output_shapes=[
#                     (None, 56, 56, 96), (None, 28, 28, 192), (None, 14, 14, 384), (None, 7, 7, 768)],
#                 expected_output_dtypes=['float32'] * 4
#             )
#
#     def test_fp16(self):
#         mixed_precision.set_global_policy('mixed_float16')
#         layer_multi_io_test(
#             Backbone,
#             kwargs={'scales': None},
#             input_shapes=[(2, 224, 224, 3)],
#             input_dtypes=['float16'],
#             expected_output_shapes=[
#                 (None, 112, 112, 64), (None, 56, 56, 256), (None, 28, 28, 512), (None, 14, 14, 1024),
#                 (None, 7, 7, 2048)],
#             expected_output_dtypes=['float16'] * 5
#         )
#         layer_multi_io_test(
#             Backbone,
#             kwargs={'scales': [2, 8]},
#             input_shapes=[(2, 224, 224, 3)],
#             input_dtypes=['uint8'],
#             expected_output_shapes=[(None, 112, 112, 64), (None, 28, 28, 512)],
#             expected_output_dtypes=['float16'] * 2
#         )
#
#         with bbpol.policy_scope('swintiny224_none'):
#             layer_multi_io_test(
#                 Backbone,
#                 kwargs={'scales': None},
#                 input_shapes=[(2, 224, 224, 3)],
#                 input_dtypes=['float16'],
#                 expected_output_shapes=[
#                     (None, 56, 56, 96), (None, 28, 28, 192), (None, 14, 14, 384), (None, 7, 7, 768)],
#                 expected_output_dtypes=['float16'] * 4
#             )
#             layer_multi_io_test(
#                 Backbone,
#                 kwargs={'scales': None},
#                 input_shapes=[(2, 224, 224, 3)],
#                 input_dtypes=['uint8'],
#                 expected_output_shapes=[
#                     (None, 56, 56, 96), (None, 28, 28, 192), (None, 14, 14, 384), (None, 7, 7, 768)],
#                 expected_output_dtypes=['float16'] * 4
#             )
#
#     def test_policy_scope_memorize(self):
#         with bbpol.policy_scope('swintiny224_none'):
#             boneinst = Backbone()
#         boneinst.build([None, None, None, 3])
#
#         restored = Backbone.from_config(boneinst.get_config())
#         restored.build([None, None, None, 3])
#
#         self.assertEqual(len(restored.trainable_weights), len(boneinst.trainable_weights))
#         self.assertEqual(len(restored.non_trainable_weights), len(boneinst.non_trainable_weights))
#
#     def test_policy_scope_memorize_build(self):
#         with bbpol.policy_scope('swintiny224_imagenet'):
#             boneinst = Backbone()
#             boneinst.build([None, None, None, 3])
#
#         restored = Backbone.from_config(boneinst.get_config())
#         restored.build([None, None, None, 3])
#
#         self.assertEqual(len(restored.trainable_weights), len(boneinst.trainable_weights))
#         self.assertEqual(len(restored.non_trainable_weights), len(boneinst.non_trainable_weights))
#
#
# if __name__ == '__main__':
#     tf.test.main()

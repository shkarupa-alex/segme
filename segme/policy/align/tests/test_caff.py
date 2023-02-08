# import tensorflow as tf
# from keras.mixed_precision import policy as mixed_precision
# from keras.testing_infra import test_combinations, test_utils
# from segme.common.align.caff import CaffFeatureAlignment, SeFeatureSelection, GuidedFeatureSelection, \
#     ImplicitKernelPrediction
# from segme.testing_utils import layer_multi_io_test
#
#
# @test_combinations.run_all_keras_modes
# class TestCaffFeatureAlignment(test_combinations.TestCase):
#     def setUp(self):
#         super(TestCaffFeatureAlignment, self).setUp()
#         self.default_policy = mixed_precision.global_policy()
#
#     def tearDown(self):
#         super(TestCaffFeatureAlignment, self).tearDown()
#         mixed_precision.set_global_policy(self.default_policy)
#
#     def test_layer(self):
#         layer_multi_io_test(
#             CaffFeatureAlignment,
#             kwargs={'filters': 6, 'pool_size': 4, 'kernel_size': 5, 'reduce_ratio': 0.75},
#             input_shapes=[(2, 16, 16, 4), (2, 8, 8, 8)],
#             input_dtypes=['float32'] * 2,
#             expected_output_shapes=[(None, 16, 16, 6)],
#             expected_output_dtypes=['float32']
#         )
#
#     def test_fp16(self):
#         mixed_precision.set_global_policy('mixed_float16')
#         layer_multi_io_test(
#             CaffFeatureAlignment,
#             kwargs={'filters': 6, 'pool_size': 4, 'kernel_size': 5, 'reduce_ratio': 0.75},
#             input_shapes=[(2, 16, 16, 4), (2, 8, 8, 8)],
#             input_dtypes=['float16'] * 2,
#             expected_output_shapes=[(None, 16, 16, 6)],
#             expected_output_dtypes=['float16']
#         )
#
#
# @test_combinations.run_all_keras_modes
# class TestSeFeatureSelection(test_combinations.TestCase):
#     def setUp(self):
#         super(TestSeFeatureSelection, self).setUp()
#         self.default_policy = mixed_precision.global_policy()
#
#     def tearDown(self):
#         super(TestSeFeatureSelection, self).tearDown()
#         mixed_precision.set_global_policy(self.default_policy)
#
#     def test_layer(self):
#         test_utils.layer_test(
#             SeFeatureSelection,
#             kwargs={'filters': 6},
#             input_shape=(2, 16, 16, 4),
#             input_dtype='float32',
#             expected_output_shape=(None, 16, 16, 6),
#             expected_output_dtype='float32'
#         )
#
#     def test_fp16(self):
#         mixed_precision.set_global_policy('mixed_float16')
#         test_utils.layer_test(
#             SeFeatureSelection,
#             kwargs={'filters': 6},
#             input_shape=(2, 16, 16, 4),
#             input_dtype='float16',
#             expected_output_shape=(None, 16, 16, 6),
#             expected_output_dtype='float16'
#         )
#
#
# @test_combinations.run_all_keras_modes
# class TestGuidedFeatureSelection(test_combinations.TestCase):
#     def setUp(self):
#         super(TestGuidedFeatureSelection, self).setUp()
#         self.default_policy = mixed_precision.global_policy()
#
#     def tearDown(self):
#         super(TestGuidedFeatureSelection, self).tearDown()
#         mixed_precision.set_global_policy(self.default_policy)
#
#     def test_layer(self):
#         layer_multi_io_test(
#             GuidedFeatureSelection,
#             kwargs={'filters': 6, 'pool_size': 4},
#             input_shapes=[(2, 16, 16, 4), (2, 8, 8, 8)],
#             input_dtypes=['float32'] * 2,
#             expected_output_shapes=[(None, 16, 16, 6)],
#             expected_output_dtypes=['float32']
#         )
#
#     def test_fp16(self):
#         mixed_precision.set_global_policy('mixed_float16')
#         layer_multi_io_test(
#             GuidedFeatureSelection,
#             kwargs={'filters': 6, 'pool_size': 4},
#             input_shapes=[(2, 16, 16, 4), (2, 8, 8, 8)],
#             input_dtypes=['float16'] * 2,
#             expected_output_shapes=[(None, 16, 16, 6)],
#             expected_output_dtypes=['float16']
#         )
#
#
# @test_combinations.run_all_keras_modes
# class TestImplicitKernelPrediction(test_combinations.TestCase):
#     def setUp(self):
#         super(TestImplicitKernelPrediction, self).setUp()
#         self.default_policy = mixed_precision.global_policy()
#
#     def tearDown(self):
#         super(TestImplicitKernelPrediction, self).tearDown()
#         mixed_precision.set_global_policy(self.default_policy)
#
#     def test_layer(self):
#         layer_multi_io_test(
#             ImplicitKernelPrediction,
#             kwargs={'filters': 9, 'kernel_size': 3},
#             input_shapes=[(2, 16, 16, 4), (2, 8, 8, 8)],
#             input_dtypes=['float32'] * 2,
#             expected_output_shapes=[(None, 16, 16, 9)],
#             expected_output_dtypes=['float32']
#         )
#
#     def test_fp16(self):
#         mixed_precision.set_global_policy('mixed_float16')
#         layer_multi_io_test(
#             ImplicitKernelPrediction,
#             kwargs={'filters': 9, 'kernel_size': 3},
#             input_shapes=[(2, 16, 16, 4), (2, 8, 8, 8)],
#             input_dtypes=['float16'] * 2,
#             expected_output_shapes=[(None, 16, 16, 9)],
#             expected_output_dtypes=['float16']
#         )
#
#
# if __name__ == '__main__':
#     tf.test.main()

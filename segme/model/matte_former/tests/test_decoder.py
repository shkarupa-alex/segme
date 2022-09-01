# import tensorflow as tf
# from keras.testing_infra import test_combinations, test_utils
# from keras.mixed_precision import policy as mixed_precision
# from segme.model.matte_former.decoder import Block, Decoder
# from segme.testing_utils import layer_multi_io_test
#
#
# @test_combinations.run_all_keras_modes
# class TestBlock(test_combinations.TestCase):
#     def test_layer(self):
#         test_utils.layer_test(
#             Block,
#             kwargs={'filters': 4, 'stride': 1},
#             input_shape=[None, 16, 16, 4],
#             input_dtype='float32',
#             expected_output_shape=[None, 16, 16, 4],
#             expected_output_dtype='float32'
#         )
#         test_utils.layer_test(
#             Block,
#             kwargs={'filters': 4, 'stride': 2},
#             input_shape=[None, 16, 16, 8],
#             input_dtype='float32',
#             expected_output_shape=[None, 32, 32, 4],
#             expected_output_dtype='float32'
#         )
#
#
# @test_combinations.run_all_keras_modes
# class TestDecoder(test_combinations.TestCase):
#     def setUp(self):
#         super(TestDecoder, self).setUp()
#         self.default_policy = mixed_precision.global_policy()
#
#     def tearDown(self):
#         super(TestDecoder, self).tearDown()
#         mixed_precision.set_global_policy(self.default_policy)
#
#     def test_layer(self):
#         layer_multi_io_test(
#             Decoder,
#             kwargs={'filters': (256, 128, 64, 32), 'depths': (2, 3, 3, 2)},
#             input_shapes=[
#                 (2, 512, 512, 32), (2, 256, 256, 32), (2, 128, 128, 64),
#                 (2, 64, 64, 128), (2, 32, 32, 256), (2, 16, 16, 512)],
#             input_dtypes=['float32'] * 6,
#             expected_output_shapes=[(None, 512, 512, 1)] * 3,
#             expected_output_dtypes=['float32'] * 3
#         )
#
#         mixed_precision.set_global_policy('mixed_float16')
#         layer_multi_io_test(
#             Decoder,
#             kwargs={'filters': (256, 128, 64, 32), 'depths': (2, 3, 3, 2)},
#             input_shapes=[
#                 (2, 512, 512, 32), (2, 256, 256, 32), (2, 128, 128, 64),
#                 (2, 64, 64, 128), (2, 32, 32, 256), (2, 16, 16, 512)],
#             input_dtypes=['float16'] * 6,
#             expected_output_shapes=[(None, 512, 512, 1)] * 3,
#             expected_output_dtypes=['float32'] * 3
#         )
#
#
#
# if __name__ == '__main__':
#     tf.test.main()

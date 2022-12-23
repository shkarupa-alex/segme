import numpy as np
import tempfile
import tensorflow as tf
from keras import layers, models
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations, test_utils
from segme.policy.backbone.diy.coma.hsmax import HSMax, TTL
from segme.policy.backbone.diy.coma.tree import tree_21k1k
from segme.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestHSMax(test_combinations.TestCase):
    def setUp(self):
        super(TestHSMax, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestHSMax, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    # def test_layer(self):
    #     layer_multi_io_test(
    #         HSMax,
    #         kwargs={'tree': tree_21k1k(), 'label_smoothing': 0.},
    #         input_datas=[
    #             np.random.uniform(size=[4, 14615]),
    #             np.array([
    #                 'apparatus.n.01', 'soda_fountain.n.02', 'computerized_axial_tomography_scanner.n.01',
    #                 'aecium.n.01'])
    #         ],
    #         input_dtypes=['float32', 'string'],
    #         expected_output_shapes=[(None, 14615)],
    #         expected_output_dtypes=['float32']
    #     )

    def test_finite(self):
        inputs = [
            layers.Input(name='features', shape=[14615], dtype='float32'),
            layers.Input(name='labels', shape=[], dtype='string')]
        outputs = HSMax(tree_21k1k(), name='hsmax', label_smoothing=0.1)(inputs)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile()

        outputs = model([
            np.random.uniform(size=[4, 14615]).astype('float32'),
            np.array([
                'apparatus.n.01', 'soda_fountain.n.02', 'computerized_axial_tomography_scanner.n.01', 'aecium.n.01'])
        ])
        outputs = self.evaluate(outputs)
        self.assertTrue(np.isfinite(outputs).all())

#     def test_serializable(self):
#         inputs = [
#             layers.Input(name='features', shape=[14615], dtype='float32'),
#             layers.Input(name='labels', shape=[], dtype='string')]
#         outputs = HSMax(tree_21k1k(), name='hsmax')(inputs)
#         model1 = models.Model(inputs=inputs, outputs=outputs)
#         model1.compile()
#
#         features = np.random.uniform(size=[4, 14615]).astype('float32')
#         labels = np.array([
#             'apparatus.n.01', 'soda_fountain.n.02', 'computerized_axial_tomography_scanner.n.01', 'aecium.n.01'])
#
#         results1 = model1([features, labels])
#         results1 = self.evaluate(results1)
#         history1 = model1.evaluate([features, labels])
#
#         with tempfile.TemporaryDirectory() as tmpdir:
#             model1.save(tmpdir)
#             model2 = models.load_model(tmpdir)
#
#         results2 = model2([features, labels])
#         results2 = self.evaluate(results2)
#         history2 = model2.evaluate([features, labels])
#
#         self.assertAllClose(results1, results2)
#         self.assertListEqual(history1, history2)
#
#     def test_fp16(self):
#         mixed_precision.set_global_policy('mixed_float16')
#         layer_multi_io_test(
#             HSMax,
#             kwargs={'tree': tree_21k1k(), 'label_smoothing': .1},
#             input_datas=[
#                 np.random.uniform(size=[4, 14615]),
#                 np.array([
#                     'apparatus.n.01', 'soda_fountain.n.02', 'computerized_axial_tomography_scanner.n.01',
#                     'aecium.n.01'])
#             ],
#             input_dtypes=['float16', 'string'],
#             expected_output_shapes=[(None, 14615)],
#             expected_output_dtypes=['float16']
#         )
#
#
# @test_combinations.run_all_keras_modes
# class TestTTL(test_combinations.TestCase):
#     def setUp(self):
#         super(TestTTL, self).setUp()
#         self.default_policy = mixed_precision.global_policy()
#
#     def tearDown(self):
#         super(TestTTL, self).tearDown()
#         mixed_precision.set_global_policy(self.default_policy)
#
#     def test_layer(self):
#         test_utils.layer_test(
#             TTL,
#             kwargs={'vocabulary': ['a,b', 'c'], 'oov_check': False},
#             input_data=np.array([['a', 'b', 'c', 'd']]),
#             input_dtype='string',
#             expected_output_shape=[None, 4],
#             expected_output_dtype='int64'
#         )
#
#     def test_value(self):
#         layer = TTL(['a,b', 'c'], oov_check=False)
#
#         result = layer(np.array([['a', 'b', 'c', 'd', 'e']]))
#         result = self.evaluate(result).tolist()
#
#         self.assertListEqual([[0, 0, 1, -1, -1]], result)
#         self.assertListEqual(['a,b', 'c'], layer.get_vocabulary())
#
#     def test_fp16(self):
#         mixed_precision.set_global_policy('mixed_float16')
#         test_utils.layer_test(
#             TTL,
#             kwargs={'vocabulary': ['a,b', 'c'], 'oov_check': False},
#             input_data=np.array([['a', 'b', 'c']]),
#             input_dtype='string',
#             expected_output_shape=[None, 3],
#             expected_output_dtype='int64'
#         )


if __name__ == '__main__':
    tf.test.main()

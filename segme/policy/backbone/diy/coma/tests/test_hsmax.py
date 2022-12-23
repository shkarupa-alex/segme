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

    def test_layer(self):
        layer_multi_io_test(
            HSMax,
            kwargs={'tree': tree_21k1k(), 'label_smoothing': 0.},
            input_datas=[
                np.random.uniform(size=[4, 14607]),
                np.array([1539, 15920, 5130, 1101])],
            input_dtypes=['float32', 'int64'],
            expected_output_shapes=[(None, 14607)],
            expected_output_dtypes=['float32']
        )

    def test_serializable(self):
        inputs = [
            layers.Input(name='features', shape=[14607], dtype='float32'),
            layers.Input(name='labels', shape=[], dtype='int64')]
        outputs = HSMax(tree_21k1k(), name='hsmax')(inputs)
        model1 = models.Model(inputs=inputs, outputs=outputs)
        model1.compile()

        features = np.random.uniform(size=[4, 14607]).astype('float32')
        labels = np.array([1539, 15920, 5130, 1101])

        results1 = model1([features, labels])
        results1 = self.evaluate(results1)
        history1 = model1.evaluate([features, labels])

        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save(tmpdir)
            model2 = models.load_model(tmpdir)

        results2 = model2([features, labels])
        results2 = self.evaluate(results2)
        history2 = model2.evaluate([features, labels])

        self.assertAllClose(results1, results2)
        self.assertListEqual(history1, history2)

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            HSMax,
            kwargs={'tree': tree_21k1k(), 'label_smoothing': .1},
            input_datas=[
                np.random.uniform(size=[4, 14607]),
                np.array([1539, 15920, 5130, 1101])
            ],
            input_dtypes=['float16', 'int64'],
            expected_output_shapes=[(None, 14607)],
            expected_output_dtypes=['float16']
        )


@test_combinations.run_all_keras_modes
class TestTTL(test_combinations.TestCase):
    def setUp(self):
        super(TestTTL, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestTTL, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            TTL,
            kwargs={'vocabulary': [[0, 1], [2]], 'oov_check': False},
            input_data=np.array([[0, 1, 2, 3]]),
            input_dtype='int64',
            expected_output_shape=[None, 4],
            expected_output_dtype='int64'
        )

    def test_value(self):
        layer = TTL([[0, 1], [2]], oov_check=False)

        result = layer(np.array([[0, 1, 2, 3, 4]]))
        result = self.evaluate(result).tolist()

        self.assertListEqual([[0, 0, 1, -1, -1]], result)
        self.assertListEqual([[0, 1], [2]], layer.get_vocabulary())

    def test_oov(self):
        layer = TTL([[0, 1], [2]], oov_check=True)
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, 'inputs should be in vocabulary.+OOV values.+3 4'):
            layer(np.array([[0, 1, 2, 3, 4]]))

    def test_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            TTL,
            kwargs={'vocabulary': [[0, 1], [2]], 'oov_check': False},
            input_data=np.array([[0, 1, 2]]),
            input_dtype='int64',
            expected_output_shape=[None, 3],
            expected_output_dtype='int64'
        )


if __name__ == '__main__':
    tf.test.main()

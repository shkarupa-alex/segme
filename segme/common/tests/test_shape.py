import tensorflow as tf
from keras.src import layers

from segme.common.shape import get_shape


class TestGetShape(tf.test.TestCase):
    def test_static(self):
        inputs = tf.zeros([2, 16, 8, 3])

        result, fully_defined = get_shape(inputs)
        self.assertTrue(fully_defined)
        self.assertListEqual(result, [2, 16, 8, 3])

        result, fully_defined = get_shape(inputs, axis=[1, 2])
        self.assertTrue(fully_defined)
        self.assertListEqual(result, [16, 8])

        result, fully_defined = get_shape(inputs, axis=[1, 2], dtype="int64")
        self.assertTrue(fully_defined)
        self.assertListEqual(result, [16, 8])

        result, fully_defined = get_shape(inputs, axis=[1, 2], dtype="float32")
        self.assertTrue(fully_defined)
        self.assertListEqual(result, [16.0, 8.0])

    def test_mixed(self):
        inputs = layers.Input(shape=[16, None, 3])

        result, fully_defined = get_shape(inputs, axis=[1, 2])
        self.assertFalse(fully_defined)
        self.assertEqual(result[0], 16)
        self.assertTrue(tf.is_tensor(result[1]))

    def test_dynamic(self):
        inputs = layers.Input(shape=[16, 8, 3])

        result, fully_defined = get_shape(inputs)
        self.assertFalse(fully_defined)
        self.assertTrue(tf.is_tensor(result[0]))
        self.assertListEqual(result[1:], [16, 8, 3])

        result, fully_defined = get_shape(inputs, axis=[1, 2])
        self.assertTrue(fully_defined)
        self.assertListEqual(result, [16, 8])

        result, fully_defined = get_shape(inputs, axis=[1, 2], dtype="int64")
        self.assertTrue(fully_defined)
        self.assertListEqual(result, [16, 8])

        result, fully_defined = get_shape(inputs, axis=[1, 2], dtype="float32")
        self.assertTrue(fully_defined)
        self.assertListEqual(result, [16.0, 8.0])

    def test_sparse(self):
        inputs = layers.Input(shape=[16, 8, 3], sparse=True)

        result, fully_defined = get_shape(inputs)
        self.assertFalse(fully_defined)
        self.assertTrue(tf.is_tensor(result[0]))
        self.assertListEqual(result[1:], [16, 8, 3])

        result, fully_defined = get_shape(inputs, axis=[1, 2])
        self.assertTrue(fully_defined)
        self.assertListEqual(result, [16, 8])

        result, fully_defined = get_shape(inputs, axis=[1, 2], dtype="int64")
        self.assertTrue(fully_defined)
        self.assertListEqual(result, [16, 8])

        result, fully_defined = get_shape(inputs, axis=[1, 2], dtype="float32")
        self.assertTrue(fully_defined)
        self.assertListEqual(result, [16.0, 8.0])

    def test_ragged(self):
        inputs = layers.Input(shape=[16, 8, 3], ragged=True)

        result, fully_defined = get_shape(inputs)
        self.assertFalse(fully_defined)
        self.assertTrue(tf.is_tensor(result[0]))
        self.assertListEqual(result[1:], [16, 8, 3])

        result, fully_defined = get_shape(inputs, axis=[1, 2])
        self.assertTrue(fully_defined)
        self.assertListEqual(result, [16, 8])

        result, fully_defined = get_shape(inputs, axis=[1, 2], dtype="int64")
        self.assertTrue(fully_defined)
        self.assertListEqual(result, [16, 8])

        result, fully_defined = get_shape(inputs, axis=[1, 2], dtype="float32")
        self.assertTrue(fully_defined)
        self.assertListEqual(result, [16.0, 8.0])

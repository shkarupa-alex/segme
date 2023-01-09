import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.python.framework import random_seed as random_seed_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from segme.common.patchxla import extract_patches_xla


class ExtractImagePatchesGradTest(tf.test.TestCase, parameterized.TestCase):
    _TEST_CASES = [
        {
            'in_shape': [2, 5, 5, 3],
            'ksizes': [1, 1, 1, 1],
            'strides': [1, 2, 3, 1],
            'rates': [1, 1, 1, 1],
        },
        {
            'in_shape': [2, 7, 7, 3],
            'ksizes': [1, 3, 3, 1],
            'strides': [1, 1, 1, 1],
            'rates': [1, 1, 1, 1],
        },
        {
            'in_shape': [2, 8, 7, 3],
            'ksizes': [1, 2, 2, 1],
            'strides': [1, 1, 1, 1],
            'rates': [1, 1, 1, 1],
        },
        {
            'in_shape': [2, 7, 8, 3],
            'ksizes': [1, 3, 2, 1],
            'strides': [1, 4, 3, 1],
            'rates': [1, 1, 1, 1],
        },
        {
            'in_shape': [1, 15, 20, 3],
            'ksizes': [1, 4, 3, 1],
            'strides': [1, 1, 1, 1],
            'rates': [1, 2, 4, 1],
        },
        {
            'in_shape': [2, 7, 8, 1],
            'ksizes': [1, 3, 2, 1],
            'strides': [1, 3, 2, 1],
            'rates': [1, 2, 2, 1],
        },
        {
            'in_shape': [2, 8, 9, 4],
            'ksizes': [1, 2, 2, 1],
            'strides': [1, 4, 2, 1],
            'rates': [1, 3, 2, 1],
        },
    ]

    def test_gradient(self):
        random_seed = 42
        random_seed_lib.set_random_seed(random_seed)

        with self.cached_session():
            for test_case in self._TEST_CASES:
                np.random.seed(random_seed)
                in_shape = test_case['in_shape']
                in_val = tf.constant(np.random.random(in_shape), dtype='float32')
                ksizes = tuple(test_case['ksizes'])
                strides = tuple(test_case['strides'])
                rates = tuple(test_case['rates'])

                for padding in ['VALID', 'SAME']:
                    def extract(in_val, ksizes=ksizes, strides=strides, rates=rates, padding=padding):
                        return extract_patches_xla(in_val, ksizes, strides, rates, padding)

                    err = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(extract, [in_val]))
                    self.assertLess(err, 1e-4)

    @parameterized.parameters(set((True, tf.executing_eagerly())))
    def test_construct_gradient_with_large_images(self, use_tape):
        with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
            batch_size = 4
            height = 512
            width = 512
            ksize = 5
            shape = (batch_size, height, width, 1)

            images = tf.Variable(np.random.uniform(size=np.prod(shape)).reshape(shape), name='inputs')
            tape.watch(images)
            patches = extract_patches_xla(
                images, sizes=[1, ksize, ksize, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
            gradients = tape.gradient(patches, images)
            self.assertIsNotNone(gradients)

    def _variable_shape_gradient(self, test_shape_pattern):
        with tf.Graph().as_default():
            random_seed = 42
            random_seed_lib.set_random_seed(random_seed)

            with self.test_session():
                for test_case in self._TEST_CASES:
                    np.random.seed(random_seed)
                    in_shape = test_case['in_shape']
                    test_shape = [x if x is None else y for x, y in zip(test_shape_pattern, in_shape)]
                    in_val = tf.compat.v1.placeholder(shape=test_shape, dtype='float32')

                    feed_dict = {in_val: np.random.random(in_shape)}
                    for padding in ['VALID', 'SAME']:
                        out_val = extract_patches_xla(
                            in_val, test_case['ksizes'], test_case['strides'], test_case['rates'], padding)
                        out_val_tmp = out_val.eval(feed_dict=feed_dict)
                        out_shape = out_val_tmp.shape

                        err = gradient_checker.compute_gradient_error(in_val, in_shape, out_val, out_shape)
                        self.assertLess(err, 1e-4)

    def test_bxxc_gradient(self):
        self._variable_shape_gradient([-1, None, None, -1])

    def test_xhwx_gradient(self):
        self._variable_shape_gradient([None, -1, -1, None])

    def test_bhwc_gradient(self):
        self._variable_shape_gradient([-1, -1, -1, -1])

    def test_all_none_gradient(self):
        self._variable_shape_gradient([None, None, None, None])


if __name__ == '__main__':
    tf.test.main()

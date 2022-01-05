import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from ..aug import augment_foreground, augment_alpha, augment_trimap


@test_util.run_all_in_graph_and_eager_modes
class TestAugmentForeground(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.foreground = np.random.uniform(0., 255., (2, 16, 16, 3)).astype('uint8')

    def test_no_aug(self):
        foreground = augment_foreground(self.foreground, mix_prob=0., inv_prob=0.)
        self.assertListEqual(foreground.shape.as_list(), list(self.foreground.shape))

        foreground = self.evaluate(foreground)
        self.assertDTypeEqual(foreground, 'uint8')
        self.assertTupleEqual(foreground.shape, self.foreground.shape)
        self.assertAllEqual(self.foreground, foreground)

    def test_aug_mix(self):
        foreground = augment_foreground(self.foreground, mix_prob=1., inv_prob=0.)
        foreground = self.evaluate(foreground)
        self.assertNotAllEqual(self.foreground, foreground)

    def test_aug_inv(self):
        foreground = augment_foreground(self.foreground, mix_prob=0., inv_prob=1.)
        foreground = self.evaluate(foreground)
        self.assertAllEqual(self.foreground, (255. - foreground.astype('float32')).astype('uint8'))


@test_util.run_all_in_graph_and_eager_modes
class TestAugmentAlpha(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.alpha = np.random.uniform(0., 255., (2, 16, 16, 1)).astype('uint8')

    def test_no_aug(self):
        alpha = augment_alpha(self.alpha, prob=0.)
        self.assertListEqual(alpha.shape.as_list(), list(self.alpha.shape))

        alpha = self.evaluate(alpha)
        self.assertDTypeEqual(alpha, 'uint8')
        self.assertTupleEqual(alpha.shape, self.alpha.shape)
        self.assertAllEqual(self.alpha, alpha)

    def test_aug(self):
        alpha = augment_alpha(self.alpha, prob=1.)
        alpha = self.evaluate(alpha)
        self.assertNotAllEqual(self.alpha, alpha)


@test_util.run_all_in_graph_and_eager_modes
class TestAugmentTrimap(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        trimap = np.zeros((2, 16, 16, 1), 'uint8')
        trimap[:, :4, ...] = 128
        trimap[:, -4:, ...] = 255
        self.trimap = trimap

    def test_no_aug(self):
        trimap = augment_trimap(self.trimap, prob=0.)
        trimap = self.evaluate(trimap)
        self.assertAllEqual(self.trimap, trimap)

    def test_aug(self):
        trimap = augment_trimap(self.trimap, prob=1.)
        trimap = self.evaluate(trimap)
        self.assertNotAllEqual(self.trimap, trimap)
        self.assertSetEqual(set(trimap.ravel()), {0, 128})


if __name__ == '__main__':
    tf.test.main()

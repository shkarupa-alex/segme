import tensorflow as tf
from tensorflow.python.framework import test_util
from segme.utils.common import rand_augment_full
from segme.utils.common.augs.tests.testing_utils import aug_samples


@test_util.run_all_in_graph_and_eager_modes
class TestRandAugment(tf.test.TestCase):
    def test_full(self):
        images, masks = aug_samples('autocontrast')
        masks = masks[..., :1]
        weights = tf.ones_like(masks, 'float32')

        images_, [masks_], weights_ = rand_augment_full(images, [masks], weights, magnitude=1.)

        self.assertEqual(6, images_.shape[0])
        self.assertEqual(3, images_.shape[-1])
        self.assertDTypeEqual(images_, images.dtype)

        self.assertEqual(6, masks_.shape[0])
        self.assertEqual(1, masks_.shape[-1])
        self.assertDTypeEqual(masks_, masks_.dtype)

        self.assertEqual(6, weights_.shape[0])
        self.assertEqual(1, weights_.shape[-1])
        self.assertDTypeEqual(weights_, weights.dtype)


if __name__ == '__main__':
    tf.test.main()

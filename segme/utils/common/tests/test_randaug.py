import tensorflow as tf

from segme.utils.common.randaug import _AUG_FUNC
from segme.utils.common.randaug import rand_augment_full


class TestRandAugment(tf.test.TestCase):
    def test_known_square(self):
        images = tf.random.uniform([8, 32, 32, 3])
        masks = tf.random.uniform([8, 32, 32, 2], maxval=4, dtype="int32")
        weights = tf.random.uniform([8, 32, 32, 1], dtype="float32")

        images_, [masks_], weights_ = rand_augment_full(
            images, [masks], weights, levels=len(_AUG_FUNC), magnitude=1.0
        )

        self.assertShapeEqual(images_, images)
        self.assertDTypeEqual(images_, images.dtype)

        self.assertShapeEqual(masks_, masks)
        self.assertDTypeEqual(masks_, masks_.dtype)

        self.assertShapeEqual(weights_, weights)
        self.assertDTypeEqual(weights_, weights.dtype)

    def test_known_non_square_no_rotate(self):
        images = tf.random.uniform([8, 32, 64, 3])
        masks = tf.random.uniform([8, 32, 64, 2], maxval=4, dtype="int32")
        weights = tf.random.uniform([8, 32, 64, 1], dtype="float32")

        ops = list(set(_AUG_FUNC.keys()) - {"RotateCCW", "RotateCW"})
        images_, [masks_], weights_ = rand_augment_full(
            images, [masks], weights, levels=len(ops), magnitude=1.0, ops=ops
        )

        self.assertShapeEqual(images_, images)
        self.assertDTypeEqual(images_, images.dtype)

        self.assertShapeEqual(masks_, masks)
        self.assertDTypeEqual(masks_, masks.dtype)

        self.assertShapeEqual(weights_, weights)
        self.assertDTypeEqual(weights_, weights.dtype)

    def test_known_non_square_rotate(self):
        images = tf.random.uniform([8, 32, 64, 3])
        masks = tf.random.uniform([8, 32, 64, 2], maxval=4, dtype="int32")
        weights = tf.random.uniform([8, 32, 64, 1], dtype="float32")

        images_, [masks_], weights_ = rand_augment_full(
            images, [masks], weights, levels=len(_AUG_FUNC), magnitude=1.0
        )

        self.assertTrue(
            images_.shape
            in [
                images.shape,
                (images.shape[0], None, None, images.shape[3]),
                (
                    images.shape[0],
                    images.shape[2],
                    images.shape[1],
                    images.shape[3],
                ),
            ]
        )
        self.assertDTypeEqual(images_, images.dtype)

        self.assertTrue(
            masks_.shape
            in [
                masks.shape,
                (masks.shape[0], None, None, masks.shape[3]),
                (
                    masks.shape[0],
                    masks.shape[2],
                    masks.shape[1],
                    masks.shape[3],
                ),
            ]
        )
        self.assertDTypeEqual(masks_, masks.dtype)

        self.assertTrue(
            weights_.shape
            in [
                weights.shape,
                (weights.shape[0], None, None, weights.shape[3]),
                (
                    weights.shape[0],
                    weights.shape[2],
                    weights.shape[1],
                    weights.shape[3],
                ),
            ]
        )
        self.assertDTypeEqual(weights_, weights.dtype)

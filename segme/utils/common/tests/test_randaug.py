from keras.src import ops
from keras.src import testing

from segme.utils.common.randaug import _AUG_FUNC
from segme.utils.common.randaug import rand_augment_full


class TestRandAugment(testing.TestCase):
    def test_known_square(self):
        images = ops.random.uniform([8, 32, 32, 3])
        masks = ops.random.uniform([8, 32, 32, 2], maxval=4, dtype="int32")
        weights = ops.random.uniform([8, 32, 32, 1], dtype="float32")

        images_, [masks_], weights_ = rand_augment_full(
            images, [masks], weights, levels=len(_AUG_FUNC), magnitude=1.0
        )

        self.assertEqual(images_.shape, images.shape)
        self.assertEqual(images_.dtype, images.dtype)

        self.assertEqual(masks_.shape, masks.shape)
        self.assertEqual(masks_.dtype, masks_.dtype)

        self.assertEqual(weights_.shape, weights.shape)
        self.assertEqual(weights_.dtype, weights.dtype)

    def test_known_non_square_no_rotate(self):
        images = ops.random.uniform([8, 32, 64, 3])
        masks = ops.random.uniform([8, 32, 64, 2], maxval=4, dtype="int32")
        weights = ops.random.uniform([8, 32, 64, 1], dtype="float32")

        operations = list(set(_AUG_FUNC.keys()) - {"RotateCCW", "RotateCW"})
        images_, [masks_], weights_ = rand_augment_full(
            images,
            [masks],
            weights,
            levels=len(operations),
            magnitude=1.0,
            operations=operations,
        )

        self.assertEqual(images_.shape, images.shape)
        self.assertEqual(images_.dtype, images.dtype)

        self.assertEqual(masks_.shape, masks.shape)
        self.assertEqual(masks_.dtype, masks.dtype)

        self.assertEqual(weights_.shape, weights.shape)
        self.assertEqual(weights_.dtype, weights.dtype)

    def test_known_non_square_rotate(self):
        images = ops.random.uniform([8, 32, 64, 3])
        masks = ops.random.uniform([8, 32, 64, 2], maxval=4, dtype="int32")
        weights = ops.random.uniform([8, 32, 64, 1], dtype="float32")

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
        self.assertEqual(images_.dtype, images.dtype)

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
        self.assertEqual(masks_.dtype, masks.dtype)

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
        self.assertEqual(weights_.dtype, weights.dtype)

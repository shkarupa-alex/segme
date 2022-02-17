import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from ..aug import augment_onthefly


@test_util.run_all_in_graph_and_eager_modes
class TestAugmentOnTheFly(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.images = [np.random.uniform(0., 255., (8, 16, 16, 3)).astype('uint8')]
        self.masks = [np.random.uniform(0., 255., (8, 16, 16, 1)).astype('uint8'),
                      np.random.uniform(0., 255., (8, 16, 16, 1)).astype('uint8')]

    def test_no_aug(self):
        images, masks = augment_onthefly(
            self.images, self.masks, hflip_prob=0., vflip_prob=0., rotate_prob=0., brightness_prob=0., contrast_prob=0.,
            hue_prob=0., saturation_prob=0.)
        for i in range(len(self.images)):
            image = self.evaluate(images[i])
            self.assertAllEqual(self.images[i], image)
        for i in range(len(self.masks)):
            mask = self.evaluate(masks[i])
            self.assertAllEqual(self.masks[i], mask)

    def test_aug(self):
        kwargs = {'hflip_prob': 0., 'vflip_prob': 0., 'rotate_prob': 0., 'brightness_prob': 0., 'contrast_prob': 0.,
                  'hue_prob': 0., 'saturation_prob': 0.}

        for aug in kwargs.keys():
            kwargs_ = dict(kwargs)
            kwargs_[aug] = 1.

            images, masks = augment_onthefly(self.images, self.masks, **kwargs_)

            for i in range(len(self.images)):
                image = self.evaluate(images[i])
                self.assertNotAllEqual(self.images[i], image, tf.executing_eagerly())

            if aug in {'hflip_prob', 'vflip_prob', 'rotate_prob'}:
                for i in range(len(self.masks)):
                    mask = self.evaluate(masks[i])
                    self.assertNotAllEqual(self.masks[i], mask)


if __name__ == '__main__':
    tf.test.main()

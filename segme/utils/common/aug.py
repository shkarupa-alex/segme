import functools
import numpy as np
import tensorflow as tf


def augment_onthefly(images, masks, hflip_prob=0.8, vflip_prob=0.6, rotate_prob=0.7, brightness_prob=0.2,
                     brightness_delta=0.2, contrast_prob=0.2, contrast_lower=0.8, contrast_upper=0.99, hue_prob=0.2,
                     hue_delta=0.2, saturation_prob=0.2, saturation_lower=0.7, saturation_upper=0.99, name=None):
    with tf.name_scope(name or 'augment_onthefly'):
        if not isinstance(images, list):
            raise ValueError('Expecting `images` to be a list.')

        images_ = []
        for i in range(len(images)):
            images_.append(tf.convert_to_tensor(images[i], 'uint8'))

            if 4 != images_[i].shape.rank:
                raise ValueError('Expecting `images` items rank to be 4.')

            if 'uint8' != images_[i].dtype:
                raise ValueError('Expecting `images` items dtype to be `uint8`.')

            images_[i] = tf.cast(images_[i], 'float32') / 255.

        if not isinstance(masks, list):
            raise ValueError('Expecting `masks` to be a list.')

        masks_ = []
        for i in range(len(masks)):
            mask_dtype = None
            if isinstance(masks[i], np.ndarray):
                mask_dtype = str(masks[i].dtype)

            masks_.append(tf.convert_to_tensor(masks[i], dtype=mask_dtype))

            if 4 != masks_[i].shape.rank:
                raise ValueError('Expecting `masks` items rank to be 4.')

        seed = tf.cast(tf.random.uniform([2], 0., 1024.), 'int32')
        probs = tf.convert_to_tensor([
            hflip_prob, vflip_prob, rotate_prob, brightness_prob, contrast_prob, hue_prob, saturation_prob], 'float32')
        apply = tf.random.uniform(probs.shape, 0., 1.) < probs
        hflip_apply, vflip_apply, rotate_apply, brightness_apply, contrast_apply, hue_apply, saturation_apply = \
            tf.unstack(apply)

        for i in range(len(images_)):
            images_[i] = tf.cond(
                hflip_apply,
                lambda: tf.image.stateless_random_flip_left_right(images_[i], seed=seed),
                lambda: images_[i])
        for i in range(len(masks_)):
            masks_[i] = tf.cond(
                hflip_apply,
                lambda: tf.image.stateless_random_flip_left_right(masks_[i], seed=seed),
                lambda: masks_[i])

        for i in range(len(images_)):
            images_[i] = tf.cond(
                vflip_apply,
                lambda: tf.image.stateless_random_flip_up_down(images_[i], seed=seed),
                lambda: images_[i])
        for i in range(len(masks_)):
            masks_[i] = tf.cond(
                vflip_apply,
                lambda: tf.image.stateless_random_flip_up_down(masks_[i], seed=seed),
                lambda: masks_[i])

        for i in range(len(images_)):
            images_[i] = tf.cond(
                rotate_apply,
                lambda: stateless_random_rotate_90(images_[i], seed=seed),
                lambda: images_[i])
        for i in range(len(masks_)):
            masks_[i] = tf.cond(
                rotate_apply,
                lambda: stateless_random_rotate_90(masks_[i], seed=seed),
                lambda: masks_[i])

        for i in range(len(images_)):
            images_[i] = tf.cond(
                brightness_apply,
                lambda: tf.image.stateless_random_brightness(images_[i], max_delta=brightness_delta, seed=seed),
                lambda: images_[i])

        for i in range(len(images_)):
            images_[i] = tf.cond(
                contrast_apply,
                lambda: tf.image.stateless_random_contrast(
                    images_[i], lower=contrast_lower, upper=contrast_upper, seed=seed),
                lambda: images_[i])

        for i in range(len(images_)):
            images_[i] = tf.cond(
                hue_apply,
                lambda: tf.image.stateless_random_hue(images_[i], max_delta=hue_delta, seed=seed),
                lambda: images_[i])

        for i in range(len(images_)):
            images_[i] = tf.cond(
                saturation_apply,
                lambda: tf.image.stateless_random_saturation(
                    images_[i], lower=saturation_lower, upper=saturation_upper, seed=seed),
                lambda: images_[i])

        for i in range(len(images_)):
            images_[i] = tf.cast(tf.round(images_[i] * 255.), 'uint8')

        # Batched images not supported
        # apply_jpeg = apply[5] < 0.1
        # for i in range(len(images)):
        #     images[i] = tf.cond(
        #         apply_jpeg,
        #         lambda: tf.image.stateless_random_jpeg_quality(
        #             images[i], min_jpeg_quality=75, max_jpeg_quality=99, seed=seed),
        #         lambda: images[i])

        return images_, masks_


def stateless_random_rotate_90(image, seed):
    random_func = functools.partial(tf.random.stateless_uniform, seed=seed)

    with tf.name_scope('stateless_random_rotate_90'):
        image = tf.convert_to_tensor(image, name='image')
        batch, height, width, _ = tf.unstack(tf.shape(image))

        assert_rank = tf.assert_rank(image, 4)
        assert_square = tf.assert_equal(height, width)

        with tf.control_dependencies([assert_rank, assert_square]):
            uniform_random = random_func(shape=[batch, 1, 1, 1], minval=0, maxval=1.0)
            flip_cw = uniform_random < 1. / 3.
            flip_ccw = uniform_random > 2. / 3.
            flip_no = (~flip_cw) & (~flip_ccw)

            image_t = tf.transpose(image, [0, 2, 1, 3])
            image_cw = tf.reverse(image_t, [2])
            image_ccw = tf.reverse(image_t, [1])

            return image * tf.cast(flip_no, image.dtype) + \
                   image_cw * tf.cast(flip_cw, image.dtype) + \
                   image_ccw * tf.cast(flip_ccw, image.dtype)

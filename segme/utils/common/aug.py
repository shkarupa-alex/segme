import functools
import numpy as np
import tensorflow as tf


def augment_onthefly(image, masks, hflip_prob=0.5, vflip_prob=0.3, rotate_prob=0.3, brightness_prob=0.2,
                     brightness_delta=0.2, contrast_prob=0.2, contrast_lower=0.8, contrast_upper=0.99, hue_prob=0.2,
                     hue_delta=0.2, saturation_prob=0.2, saturation_lower=0.7, saturation_upper=0.99, mix_prob=0.3,
                     mix_max=0.3, shuffle_prob=0.2, name=None):
    if hflip_prob > 0.5 or vflip_prob > 0.5:
        raise ValueError('Horizontal and vertical flip with probability above 50% not supported')
    if rotate_prob > 2 / 3:
        raise ValueError('Rotate with probability above 66% not supported')

    with tf.name_scope(name or 'augment_onthefly'):
        image_ = tf.convert_to_tensor(image)
        if 4 != image_.shape.rank:
            raise ValueError('Expecting `image` rank to be 4.')

        orig_dtype = image.dtype
        image_ = image if orig_dtype in {tf.float16, tf.float32} else tf.image.convert_image_dtype(image, 'float32')

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
            hflip_prob * 2., vflip_prob * 2., rotate_prob * 3 / 2,
            brightness_prob, contrast_prob, hue_prob, saturation_prob, mix_prob, shuffle_prob], 'float32')
        apply = tf.random.uniform(probs.shape, 0., 1.) < probs
        (hflip_apply, vflip_apply, rotate_apply, brightness_apply, contrast_apply, hue_apply, saturation_apply,
         mix_apply, shuffle_apply) = tf.unstack(apply)

        image_ = tf.cond(
            hflip_apply,
            lambda: tf.image.stateless_random_flip_left_right(image_, seed=seed),
            lambda: tf.identity(image_))
        for i in range(len(masks_)):
            masks_[i] = tf.cond(
                hflip_apply,
                lambda: tf.image.stateless_random_flip_left_right(masks_[i], seed=seed),
                lambda: tf.identity(masks_[i]))

        image_ = tf.cond(
            vflip_apply,
            lambda: tf.image.stateless_random_flip_up_down(image_, seed=seed),
            lambda: tf.identity(image_))
        for i in range(len(masks_)):
            masks_[i] = tf.cond(
                vflip_apply,
                lambda: tf.image.stateless_random_flip_up_down(masks_[i], seed=seed),
                lambda: tf.identity(masks_[i]))

        image_ = tf.cond(
            rotate_apply,
            lambda: stateless_random_rotate_90(image_, seed=seed),
            lambda: tf.identity(image_))
        for i in range(len(masks_)):
            masks_[i] = tf.cond(
                rotate_apply,
                lambda: stateless_random_rotate_90(masks_[i], seed=seed),
                lambda: tf.identity(masks_[i]))

        image_ = tf.cond(
            brightness_apply,
            lambda: tf.image.random_brightness(image_, max_delta=brightness_delta),
            lambda: tf.identity(image_))

        image_ = tf.cond(
            contrast_apply,
            lambda: tf.image.random_contrast(image_, lower=contrast_lower, upper=contrast_upper),
            lambda: tf.identity(image_))

        image_ = tf.cond(
            hue_apply,
            lambda: tf.image.random_hue(image_, max_delta=hue_delta),
            lambda: tf.identity(image_))

        image_ = tf.cond(
            saturation_apply,
            lambda: tf.image.random_saturation(image_, lower=saturation_lower, upper=saturation_upper),
            lambda: tf.identity(image_))

        image_ = tf.cond(
            mix_apply,
            lambda: random_color_mix(image_, mix_max=mix_max),
            lambda: tf.identity(image_))

        image_ = tf.cond(
            shuffle_apply,
            lambda: random_channel_shuffle(image_),
            lambda: tf.identity(image_))

        # Batched images not supported
        # apply_jpeg = apply[5] < 0.1
        # for i in range(len(images)):
        #     images[i] = tf.cond(
        #         apply_jpeg,
        #         lambda: tf.image.stateless_random_jpeg_quality(
        #             images[i], min_jpeg_quality=75, max_jpeg_quality=99, seed=seed),
        #         lambda: images[i])

        image_ = tf.image.convert_image_dtype(image_, orig_dtype, saturate=True)

        return image_, masks_


def stateless_random_rotate_90(image, seed):
    random_func = functools.partial(tf.random.stateless_uniform, seed=seed)

    with tf.name_scope('stateless_random_rotate_90'):
        image = tf.convert_to_tensor(image, name='image')

        assert_rank = tf.assert_rank(image, 4)
        with tf.control_dependencies([assert_rank]):
            image = tf.identity(image)

        batch, height, width, _ = tf.unstack(tf.shape(image))
        assert_square = tf.assert_equal(height, width)
        with tf.control_dependencies([assert_square]):
            image = tf.identity(image)

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


def random_color_mix(image, mix_max, seed=None):
    with tf.name_scope('random_color_mix'):
        image = tf.convert_to_tensor(image, name='image')
        batch, _, _, channel = tf.unstack(tf.shape(image))

        assert_rank = tf.assert_rank(image, 4)
        assert_channel = tf.assert_equal(channel, 3)
        with tf.control_dependencies([assert_rank, assert_channel]):
            image = tf.identity(image)

        color = tf.random.uniform(shape=[batch, 1, 1, 3], minval=0., maxval=1., seed=seed)
        weight = tf.random.uniform(shape=[batch, 1, 1, 1], minval=0., maxval=mix_max, seed=seed)

        orig_dtype = image.dtype
        image_ = image if orig_dtype in {tf.float16, tf.float32} else tf.image.convert_image_dtype(image, 'float32')

        image_ = image_ * (1. - weight) + color * weight

        return tf.image.convert_image_dtype(image_, orig_dtype, saturate=True)


def random_channel_shuffle(image, seed=None):
    with tf.name_scope('random_channel_shuffle'):
        image = tf.convert_to_tensor(image, name='image')

        assert_rank = tf.assert_rank(image, 4)
        assert_channel = tf.assert_equal(image.shape[-1], 3)
        with tf.control_dependencies([assert_rank, assert_channel]):
            image = tf.identity(image)

        batch = tf.shape(image)[0]
        switch = tf.random.uniform(shape=[batch, 1, 1, 1], minval=0., maxval=1., seed=seed)
        permutations = [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]

        image_ = tf.gather(image, permutations[0], batch_dims=-1) * tf.cast(switch <= 1 / 5, image.dtype)
        for i in range(1, 5):
            apply = tf.cast((switch > i / 5) & (switch <= (i + 1) / 5), image.dtype)
            image_ += tf.gather(image, permutations[i], batch_dims=-1) * apply

        return image_

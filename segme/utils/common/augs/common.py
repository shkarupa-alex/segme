import tensorflow as tf


def apply(image, masks, weight, prob, image_fn, mask_fn, weight_fn, name=None):
    with tf.name_scope(name or 'apply'):
        def _select_some(x, x_, s, h, w, h_, w_):
            with tf.name_scope(name or 'select_some'):
                mh = tf.maximum(h, h_)
                mw = tf.maximum(w, w_)
                x = tf.pad(x, [(0, 0), (0, mh - h), (0, mw - w), (0, 0)])
                x_ = tf.pad(x_, [(0, 0), (0, mh - h_), (0, mw - w_), (0, 0)])
                s = tf.cast(s, x.dtype)

                x = x * (1 - s) + x_ * s

                return x

        def _select_all(x, x_, s):
            with tf.name_scope(name or 'select_all'):
                r = tf.cond(s, lambda: tf.identity(x_), lambda: tf.identity(x))
                return r

        image, masks, weight = validate(image, masks, weight)
        image_ = image_fn(image)

        batch, height, width = tf.unstack(tf.shape(image)[:3])
        height_, width_ = tf.unstack(tf.shape(image_)[1:3])
        same = tf.logical_and(tf.equal(height, height_), tf.equal(width, width_))

        switch_full = tf.random.uniform([batch, 1, 1, 1]) < prob
        switch_all = switch_full[0, 0, 0, 0]
        switch_part = tf.cast(switch_full[..., :1], 'float32')

        image = tf.cond(
            same,
            lambda: _select_some(image, image_, switch_part, height, width, height_, width_),
            lambda: _select_all(image, image_, switch_all))

        if masks is not None:
            temp = []
            for mask in masks:
                mask_ = mask_fn(mask)
                mask = tf.cond(
                    same,
                    lambda: _select_some(mask, mask_, switch_part, height, width, height_, width_),
                    lambda: _select_all(mask, mask_, switch_all))
                temp.append(mask)
            masks = temp

        if weight is not None:
            weight_ = weight_fn(weight)
            weight = tf.cond(
                same,
                lambda: _select_some(weight, weight_, switch_part, height, width, height_, width_),
                lambda: _select_all(weight, weight_, switch_all))

        return image, masks, weight


def convert(image, dtype, saturate=False, name=None):
    with tf.name_scope(name or 'convert'):
        image = tf.convert_to_tensor(image, name='image')
        dtype = tf.dtypes.as_dtype(dtype)

        if image.dtype.is_floating and dtype.is_integer:
            # TODO: https://github.com/tensorflow/tensorflow/pull/54484
            image *= dtype.max
            image = tf.round(image)

            if saturate:
                return tf.saturate_cast(image, dtype)
            else:
                return tf.cast(image, dtype)

        if image.dtype.is_floating and dtype.is_floating and saturate:
            return tf.clip_by_value(image, 0., 1.)

        return tf.image.convert_image_dtype(image, dtype, saturate=saturate)


def blend(original, augmented, factor, name=None):
    with tf.name_scope(name or 'blend'):
        original, _, _ = validate(original, None, None)
        augmented, _, _ = validate(augmented, None, None)

        dtype = original.dtype
        original = convert(original, 'float32')
        augmented = convert(augmented, 'float32')

        blended = original + factor * (augmented - original)

        return convert(blended, dtype, saturate=True)


def transform(images, transforms, fill_mode='reflect', fill_value=0.0, interpolation='bilinear', name=None):
    with tf.name_scope(name or 'transform'):
        output_shape = tf.shape(images)[1:3]
        if not tf.executing_eagerly():
            output_shape_value = tf.get_static_value(output_shape)
            if output_shape_value is not None:
                output_shape = output_shape_value

        output_shape = tf.convert_to_tensor(output_shape, 'int32', name='output_shape')
        fill_value = tf.convert_to_tensor(fill_value, 'float32', name='fill_value')

        return tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            output_shape=output_shape,
            fill_value=fill_value,
            transforms=transforms,
            fill_mode=fill_mode.upper(),
            interpolation=interpolation.upper(),
        )


def wrap(image, name=None):
    with tf.name_scope(name or 'wrap'):
        image, _, _ = validate(image, None, None)

        image = tf.pad(image, [(0, 0)] * 3 + [(0, 1)], constant_values=1)

        return image


def unwrap(image, replace=None, name=None):
    with tf.name_scope(name or 'wrap'):
        image, _, _ = validate(image, None, None)

        image, mask = tf.split(image, [image.shape[-1] - 1, 1], axis=-1)

        if replace is not None:
            replace = tf.convert_to_tensor(replace, image.dtype, name='replace')
            replace, _, _ = validate(replace, None, None)
        else:
            batch = tf.shape(image)[0]
            replace = tf.random.uniform([batch, 1, 1, image.shape[-1]])
            replace = convert(replace, image.dtype, saturate=True)

        mask = tf.cast(tf.equal(mask, 1), image.dtype)
        image = image * mask + replace * (1 - mask)

        return image


def validate(image, masks, weight, name=None):
    with tf.name_scope(name or 'validate'):
        image_ = tf.convert_to_tensor(image, name='image')
        if 4 != image_.shape.rank:
            raise ValueError('Expecting `image` rank to be 4.')
        if image_.shape[-1] is None:
            raise ValueError('Expecting channel dimension of the `image` to be defined. Found `None`.')

        if masks is not None:
            masks_ = []
            for i, mask in enumerate(masks):
                mask_ = tf.convert_to_tensor(mask, dtype=mask.dtype, name=f'mask_{i}')
                if 4 != mask_.shape.rank:
                    raise ValueError('Expecting `masks` items rank to be 4.')
                if mask_.shape[-1] is None:
                    raise ValueError('Expecting channel dimension of the `mask` to be defined. Found `None`.')
                masks_.append(mask_)
        else:
            masks_ = None

        if weight is not None:
            weight_ = tf.convert_to_tensor(weight, name='weight')
            if 4 != weight_.shape.rank:
                raise ValueError('Expecting `weight` rank to be 4.')
            if weight_.shape[-1] is None:
                raise ValueError('Expecting channel dimension of the `weight` to be defined. Found `None`.')
        else:
            weight_ = None

        return image_, masks_, weight_

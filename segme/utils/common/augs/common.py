import tensorflow as tf
from tf_keras.src.utils.control_flow_util import smart_cond
from tensorflow.python.ops import data_flow_ops
from segme.common.shape import get_shape


def apply(image, masks, weight, prob, image_fn, mask_fn, weight_fn, name=None):
    def _some(original, condition, aug_fn, orig_idx, aug_idx):
        with tf.name_scope(name or 'some'):
            shape = original.shape

            augmented, original = original[condition], original[~condition]
            augmented = aug_fn(augmented)

            result = data_flow_ops.parallel_dynamic_stitch([aug_idx, orig_idx], [augmented, original])
            result.set_shape(shape)

            return result

    def _all(original, aug_fn, condition):
        with tf.name_scope(name or 'all'):
            return smart_cond(condition, lambda: aug_fn(original), lambda: tf.identity(original))

    with tf.name_scope(name or 'apply'):
        image, masks, weight = validate(image, masks, weight)

        prob = tf.convert_to_tensor(prob)
        if prob.shape.rank:
            raise ValueError('Expecting `prob` to be a scalar.')

        (batch, height, width), _ = get_shape(image, axis=[0, 1, 2])
        static_size = not tf.is_tensor(height) and not tf.is_tensor(width)
        square_size = height == width if static_size else tf.equal(height, width)

        switch_some = tf.random.uniform([batch]) < prob
        switch_all = switch_some[0]

        if static_size and square_size:
            indices = tf.range(batch, dtype='int32')
            indices_, indices = indices[switch_some], indices[~switch_some]

            image = _some(image, switch_some, image_fn, indices, indices_)

            if masks is not None:
                masks = [_some(m, switch_some, mask_fn, indices, indices_) for m in masks]

            if weight is not None:
                weight = _some(weight, switch_some, weight_fn, indices, indices_)

        elif static_size and not square_size:
            image = _all(image, image_fn, switch_all)

            if masks is not None:
                masks = [_all(m, mask_fn, switch_all) for m in masks]

            if weight is not None:
                weight = _all(weight, weight_fn, switch_all)

        else:
            indices = tf.range(batch, dtype='int32')
            indices_, indices = indices[switch_some], indices[~switch_some]

            image = smart_cond(
                square_size,
                lambda: _some(image, switch_some, image_fn, indices, indices_),
                lambda: _all(image, image_fn, switch_all))

            if masks is not None:
                temp = []
                for mask in masks:
                    mask = smart_cond(
                        square_size,
                        lambda: _some(mask, switch_some, mask_fn, indices, indices_),
                        lambda: _all(mask, mask_fn, switch_all))
                    temp.append(mask)
                masks = temp

            if weight is not None:
                weight = smart_cond(
                    square_size,
                    lambda: _some(weight, switch_some, weight_fn, indices, indices_),
                    lambda: _all(weight, weight_fn, switch_all))

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
        output_shape, _ = get_shape(images, axis=[1, 2])
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
            replace = tf.reduce_mean(image, axis=[1, 2], keepdims=True)

        image = tf.where(tf.equal(mask, 1), image, replace)

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

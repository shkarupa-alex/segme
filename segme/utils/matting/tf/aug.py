import tensorflow as tf


def augment_inverse(foreground, inv_prob=0.1, name=None):
    with tf.name_scope(name or 'augment_foreground'):
        foreground = tf.convert_to_tensor(foreground, 'uint8')

        if 4 != foreground.shape.rank:
            raise ValueError('Expecting `alpha` rank to be 4.')

        if 3 != foreground.shape[-1]:
            raise ValueError('Expecting `foreground` channels size to be 3.')

        if 'uint8' != foreground.dtype:
            raise ValueError('Expecting `foreground` dtype to be `uint8`.')

        def _ivn_transform(foreground_):
            return 255 - foreground_

        inv_apply = tf.random.uniform([], 0., 1.) < inv_prob
        foreground = tf.cond(inv_apply, lambda: _ivn_transform(foreground), lambda: foreground)

        return foreground


def augment_alpha(alpha, prob=0.1, low=0.5, high=1.5, name=None):
    with tf.name_scope(name or 'augment_alpha'):
        alpha = tf.convert_to_tensor(alpha, 'uint8')

        if 4 != alpha.shape.rank:
            raise ValueError('Expecting `alpha` rank to be 4.')

        if 1 != alpha.shape[-1]:
            raise ValueError('Expecting `alpha` channels size to be 1.')

        if 'uint8' != alpha.dtype:
            raise ValueError('Expecting `alpha` dtype to be `uint8`.')

        def _transform(alpha_, rate_):
            alpha_ = tf.cast(alpha_, 'float32') / 255.
            alpha_ = alpha_ ** (rate_ * (high - low) + low)  # ^ [low; high]
            alpha_ = tf.clip_by_value(alpha_, 0., 1.)
            alpha_ = tf.cast(tf.round(alpha_ * 255.), 'uint8')

            return alpha_

        apply, rate = tf.unstack(tf.random.uniform([2], 0., 1.))
        alpha = tf.cond(
            tf.less(apply, prob),
            lambda: _transform(alpha, rate),
            lambda: alpha)

        return alpha


def augment_trimap(trimap, prob=0.1, name=None):
    with tf.name_scope(name or 'augment_trimap'):
        trimap = tf.convert_to_tensor(trimap, 'uint8')

        if 4 != trimap.shape.rank:
            raise ValueError('Expecting `trimap` rank to be 4.')

        if 1 != trimap.shape[-1]:
            raise ValueError('Expecting `trimap` channels size to be 1.')

        if 'uint8' != trimap.dtype:
            raise ValueError('Expecting `trimap` dtype to be `uint8`.')

        fg = tf.convert_to_tensor(255, 'uint8')
        un = tf.convert_to_tensor(128, 'uint8')

        apply = tf.random.uniform((), 0., 1.)
        trimap = tf.cond(
            apply < prob,
            lambda: tf.where(tf.equal(trimap, fg), un, trimap),
            lambda: trimap)

        return trimap

import tensorflow as tf
from keras import backend
from segme.utils.matting.tf.fg import solve_fg


def compose_two(fg, alpha, solve=True, prob=0.5, name=None):
    with tf.name_scope(name or 'compose_two'):
        fg = tf.convert_to_tensor(fg, 'uint8')
        alpha = tf.convert_to_tensor(alpha, 'uint8')
        eps = tf.convert_to_tensor(backend.epsilon(), 'float32')

        if 4 != len(fg.shape):
            raise ValueError('Expecting `fg` rank to be 4.')

        if 3 != fg.shape[-1]:
            raise ValueError('Expecting `fg` channels size to be 3.')

        if 4 != len(alpha.shape):
            raise ValueError('Expecting `alpha` rank to be 4.')

        if 1 != alpha.shape[-1]:
            raise ValueError('Expecting `alpha` channels size to be 1.')

        if 'uint8' != fg.dtype or 'uint8' != alpha.dtype:
            raise ValueError('Expecting `fg` and `alpha` dtype to be `uint8`.')

        def _transform(fg01, alpha01):
            alpha01 = tf.cast(alpha01, 'float32') / 255.
            alpha0, alpha1 = tf.split(alpha01, 2, axis=0)

            # Combine fgs and alphas
            # For description see https://github.com/Yaoyi-Li/GCA-Matting/issues/12
            # and https://github.com/MarcoForte/FBA_Matting/issues/44
            delta = alpha1 * (1. - alpha0)
            alpha_ = alpha0 + delta

            accept = tf.reduce_any(alpha_ > 0., axis=[1, 2, 3]) & tf.reduce_any(alpha_ < 1., axis=[1, 2, 3])
            alpha_ = alpha_[accept]
            alpha0 = alpha0[accept]
            delta = delta[accept]

            accept_twice = tf.tile(accept, [2])
            fg01 = fg01[accept_twice]

            fg01 = tf.cast(fg01, 'float32') / 255.
            fg0, fg1 = tf.split(fg01, 2, axis=0)

            # The overlap of two 50% transparency should be 25%
            fg_ = (fg0 * alpha0 + fg1 * delta) / (alpha_ + eps)
            fg_ = tf.clip_by_value(fg_, 0., 1.)

            fg_ = tf.cast(tf.round(fg_ * 255.), 'uint8')
            alpha_ = tf.cast(tf.round(alpha_ * 255.), 'uint8')

            if solve:
                fg_ = solve_fg(fg_, alpha_)

            return fg_, alpha_

        even_batch = tf.equal(tf.math.mod(tf.shape(fg)[0], 2), 0)
        apply = tf.random.uniform((), 0., 1.) < prob

        fg, alpha = tf.cond(
            even_batch & apply,
            lambda: _transform(fg, alpha),
            lambda: (tf.identity(fg), tf.identity(alpha)))

        return fg, alpha

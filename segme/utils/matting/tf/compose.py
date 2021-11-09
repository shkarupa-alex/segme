import tensorflow as tf
from keras import backend as K
from .fbgb import solve_fgbg


def compose_two(fg, alpha, rest=None, prob=0.5, solve=False, regularization=0.005, small_size=32, small_steps=10,
                big_steps=2, grad_weight=0.1, name=None):
    with tf.name_scope(name or 'compose_two'):
        fg = tf.convert_to_tensor(fg, 'uint8')
        alpha = tf.convert_to_tensor(alpha, 'uint8')
        eps = tf.convert_to_tensor(K.epsilon(), 'float32')

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

        if rest is None:
            rest = []
        if not isinstance(rest, list):
            raise ValueError('Expecting `rest` to be a list.')

        rest_ = []
        for i in range(len(rest)):
            rest_.append(tf.convert_to_tensor(rest[i]))

            if 1 > rest_[i].shape.rank:
                raise ValueError('Expecting `rest` items rank to be at least 1.')

        def _transform(fg01, alpha01, rest01):
            alpha01 = tf.cast(alpha01, 'float32') / 255.
            alpha0, alpha1 = tf.split(alpha01, 2, axis=0)

            # Combine fgs and alphas
            # For description see https://github.com/Yaoyi-Li/GCA-Matting/issues/12
            # and https://github.com/MarcoForte/FBA_Matting/issues/44
            delta = alpha1 * (1. - alpha0)
            alpha_ = alpha0 + delta

            accept = tf.reduce_any(alpha_ > 0., axis=[1, 2, 3]) & tf.reduce_any(alpha_ < 1., axis=[1, 2, 3])
            alpha_ = alpha_[accept]
            delta = delta[accept]

            accept01 = tf.tile(accept, [2])
            accept10 = tf.concat([accept, tf.zeros_like(accept, dtype='bool')], 0)

            fg01 = fg01[accept01]
            rest01_ = [r[accept10] for r in rest01]

            fg01 = tf.cast(fg01, 'float32') / 255.
            fg0, fg1 = tf.split(fg01, 2, axis=0)

            # The overlap of two 50% transparency should be 25%
            fg_ = (fg0 * alpha0 + fg1 * delta) / (alpha_ + eps)
            fg_ = tf.clip_by_value(fg_, 0., 1.)

            fg_ = tf.cast(tf.round(fg_ * 255.), 'uint8')
            alpha_ = tf.cast(tf.round(alpha_ * 255.), 'uint8')

            if solve:
                fg_, _ = solve_fgbg(
                    fg_, alpha_, regularization=regularization, small_size=small_size, small_steps=small_steps,
                    big_steps=big_steps, grad_weight=grad_weight)

            return fg_, alpha_, rest01_

        apply = tf.random.uniform((), 0., 1.)
        fg, alpha, rest_ = tf.cond(
            tf.less(apply, prob),
            lambda: _transform(fg, alpha, rest_),
            lambda: (fg, alpha, rest_))

        if rest:
            return fg, alpha, rest_

        return fg, alpha
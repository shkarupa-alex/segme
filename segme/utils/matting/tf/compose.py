import tensorflow as tf
from keras import backend
from segme.utils.matting.tf.fg import solve_fg
from segme.utils.matting.tf.trimap import alpha_trimap
from segme.common.shape import get_shape


def compose_two(fg, alpha, solve=True, name=None):
    with tf.name_scope(name or 'compose_two'):
        fg = tf.convert_to_tensor(fg, 'uint8')
        alpha = tf.convert_to_tensor(alpha, 'uint8')

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

        (batch,), _ = get_shape(fg, axis=[0])
        batch = (batch // 2) * 2
        fg = fg[:batch]
        alpha = alpha[:batch]

        alpha = tf.cast(alpha, 'float32') / 255.
        alpha0, alpha1 = tf.split(alpha, 2, axis=0)

        # Combine fgs and alphas
        # For description see https://github.com/Yaoyi-Li/GCA-Matting/issues/12
        # and https://github.com/MarcoForte/FBA_Matting/issues/44
        delta = alpha1 * (1. - alpha0)
        alpha_ = alpha0 + delta

        accept = tf.reduce_any(alpha_ > 0., axis=[1, 2, 3]) & tf.reduce_any(alpha_ < 1., axis=[1, 2, 3])
        alpha_ = alpha_[accept]
        alpha0 = alpha0[accept]
        delta = delta[accept]

        fg = fg[tf.tile(accept, [2])]
        fg = tf.cast(fg, 'float32') / 255.
        fg0, fg1 = tf.split(fg, 2, axis=0)

        # The overlap of two 50% transparency should be 25%
        fg_ = (fg0 * alpha0 + fg1 * delta) / (alpha_ + backend.epsilon())
        fg_ = tf.clip_by_value(fg_, 0., 1.)

        fg_ = tf.cast(tf.round(fg_ * 255.), 'uint8')
        alpha_ = tf.cast(tf.round(alpha_ * 255.), 'uint8')

        if solve:
            fg_ = solve_fg(fg_, alpha_)

        return fg_, alpha_


def random_compose(fg, alpha, prob=0.5, trim=None, solve=True, iterations=4, name=None):
    with tf.name_scope(name or 'random_compose'):
        fg = tf.convert_to_tensor(fg, 'uint8')
        alpha = tf.convert_to_tensor(alpha, 'uint8')

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

        (batch, height, width), _ = get_shape(fg, axis=[0, 1, 2])
        indices = tf.range(batch)

        def _cond(comp_fg, comp_alpha):
            (batch_,), _ = get_shape(comp_fg, axis=[0])

            return batch_ < batch

        def _body(comp_fg, comp_alpha):
            curr_indices = tf.random.shuffle(indices)
            curr_fg = tf.gather(fg, curr_indices, axis=0)
            curr_alpha = tf.gather(alpha, curr_indices, axis=0)
            curr_fg, curr_alpha = compose_two(curr_fg, curr_alpha, solve=False)

            if trim is not None:
                curr_trimap = alpha_trimap(curr_alpha, trim[0])
                curr_mask = tf.cast(tf.equal(curr_trimap, 128), 'float32')
                curr_mask = tf.reduce_mean(curr_mask, axis=[1, 2, 3]) < trim[1]
                curr_fg = curr_fg[curr_mask]
                curr_alpha = curr_alpha[curr_mask]

            comp_fg = tf.concat([comp_fg, curr_fg], axis=0)
            comp_alpha = tf.concat([comp_alpha, curr_alpha], axis=0)

            return comp_fg, comp_alpha

        fg_ = tf.zeros([0, height, width, 3], dtype='uint8')
        alpha_ = tf.zeros([0, height, width, 1], dtype='uint8')
        fg_, alpha_ = tf.while_loop(
            _cond, _body, (fg_, alpha_), maximum_iterations=iterations, shape_invariants=(
                tf.TensorShape([None, None, None, 3]), tf.TensorShape([None, None, None, 1])))
        fg_ = tf.concat([fg_, fg], axis=0)[:batch]
        alpha_ = tf.concat([alpha_, alpha], axis=0)[:batch]
        fg_.set_shape(fg.shape)
        alpha_.set_shape(alpha.shape)

        apply = tf.cast(tf.random.uniform([batch, 1, 1, 1], 0., 1.) < prob, 'uint8')
        fg_ = fg_ * apply + fg * (1 - apply)
        alpha_ = alpha_ * apply + alpha * (1 - apply)

        if solve:
            fg_ = solve_fg(fg_, alpha_)

        return fg_, alpha_

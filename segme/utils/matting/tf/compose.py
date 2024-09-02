from keras.src import backend
from keras.src import ops

from segme.utils.matting.tf.fg import solve_fg
from segme.utils.matting.tf.trimap import alpha_trimap


def compose_two(fg, alpha, solve=True, name=None):
    with backend.name_scope(name or "compose_two"):
        fg = backend.convert_to_tensor(fg, "uint8")
        alpha = backend.convert_to_tensor(alpha, "uint8")

        if 4 != len(fg.shape):
            raise ValueError("Expecting `fg` rank to be 4.")

        if 3 != fg.shape[-1]:
            raise ValueError("Expecting `fg` channels size to be 3.")

        if 4 != len(alpha.shape):
            raise ValueError("Expecting `alpha` rank to be 4.")

        if 1 != alpha.shape[-1]:
            raise ValueError("Expecting `alpha` channels size to be 1.")

        if "uint8" != fg.dtype or "uint8" != alpha.dtype:
            raise ValueError("Expecting `fg` and `alpha` dtype to be `uint8`.")

        batch = ops.shape(fg)[0]
        batch = (batch // 2) * 2
        fg = fg[:batch]
        alpha = alpha[:batch]

        alpha = ops.cast(alpha, "float32") / 255.0
        alpha0, alpha1 = ops.split(alpha, 2, axis=0)

        # Combine fgs and alphas
        # For description see https://github.com/Yaoyi-Li/GCA-Matting/issues/12
        # and https://github.com/MarcoForte/FBA_Matting/issues/44
        delta = alpha1 * (1.0 - alpha0)
        alpha_ = alpha0 + delta

        accept = ops.any(alpha_ > 0.0, axis=[1, 2, 3]) & ops.any(
            alpha_ < 1.0, axis=[1, 2, 3]
        )
        alpha_ = alpha_[accept]
        alpha0 = alpha0[accept]
        delta = delta[accept]

        fg = fg[ops.tile(accept, [2])]
        fg = ops.cast(fg, "float32") / 255.0
        fg0, fg1 = ops.split(fg, 2, axis=0)

        # The overlap of two 50% transparency should be 25%
        fg_ = (fg0 * alpha0 + fg1 * delta) / (alpha_ + backend.epsilon())
        fg_ = ops.clip(fg_, 0.0, 1.0)

        fg_ = ops.cast(ops.round(fg_ * 255.0), "uint8")
        alpha_ = ops.cast(ops.round(alpha_ * 255.0), "uint8")

        if solve:
            fg_ = solve_fg(fg_, alpha_)

        return fg_, alpha_


def compose_batch(fg, alpha, trim=None, solve=True, iterations=4, name=None):
    with backend.name_scope(name or "compose_batch"):
        fg = backend.convert_to_tensor(fg, "uint8")
        alpha = backend.convert_to_tensor(alpha, "uint8")

        if 4 != len(fg.shape):
            raise ValueError("Expecting `fg` rank to be 4.")

        if 3 != fg.shape[-1]:
            raise ValueError("Expecting `fg` channels size to be 3.")

        if 4 != len(alpha.shape):
            raise ValueError("Expecting `alpha` rank to be 4.")

        if 1 != alpha.shape[-1]:
            raise ValueError("Expecting `alpha` channels size to be 1.")

        if "uint8" != fg.dtype or "uint8" != alpha.dtype:
            raise ValueError("Expecting `fg` and `alpha` dtype to be `uint8`.")

        batch, height, width, _ = ops.shape(fg)
        indices = ops.arange(batch)

        def _cond(comp_fg, _):
            batch_ = ops.shape(comp_fg)[0]

            return batch_ < batch

        def _body(comp_fg, comp_alpha):
            curr_indices = ops.random.shuffle(indices)
            curr_fg = ops.take(fg, curr_indices, axis=0)
            curr_alpha = ops.take(alpha, curr_indices, axis=0)
            curr_fg, curr_alpha = compose_two(curr_fg, curr_alpha, solve=False)

            if trim is not None:
                curr_trimap = alpha_trimap(curr_alpha, trim[0])
                curr_mask = ops.cast(ops.equal(curr_trimap, 128), "float32")
                curr_mask = ops.mean(curr_mask, axis=[1, 2, 3]) < trim[1]
                curr_fg = curr_fg[curr_mask]
                curr_alpha = curr_alpha[curr_mask]

            comp_fg = ops.concatenate([comp_fg, curr_fg], axis=0)
            comp_alpha = ops.concatenate([comp_alpha, curr_alpha], axis=0)

            return comp_fg, comp_alpha

        fg_ = ops.zeros([0, height, width, 3], dtype="uint8")
        alpha_ = ops.zeros([0, height, width, 1], dtype="uint8")
        fg_, alpha_ = ops.while_loop(
            _cond,
            _body,
            (fg_, alpha_),
            maximum_iterations=iterations,
        )
        fg_ = ops.concatenate([fg_, fg], axis=0)[:batch]
        alpha_ = ops.concatenate([alpha_, alpha], axis=0)[:batch]
        fg_.set_shape(fg.shape)
        alpha_.set_shape(alpha.shape)

        if solve:
            fg_ = solve_fg(fg_, alpha_)

        return fg_, alpha_


def random_compose(
    fg, alpha, prob=0.5, trim=None, solve=True, iterations=4, name=None
):
    with backend.name_scope(name or "random_compose"):
        fg = backend.convert_to_tensor(fg, "uint8")
        alpha = backend.convert_to_tensor(alpha, "uint8")

        if 4 != len(fg.shape):
            raise ValueError("Expecting `fg` rank to be 4.")

        if 3 != fg.shape[-1]:
            raise ValueError("Expecting `fg` channels size to be 3.")

        if 4 != len(alpha.shape):
            raise ValueError("Expecting `alpha` rank to be 4.")

        if 1 != alpha.shape[-1]:
            raise ValueError("Expecting `alpha` channels size to be 1.")

        if "uint8" != fg.dtype or "uint8" != alpha.dtype:
            raise ValueError("Expecting `fg` and `alpha` dtype to be `uint8`.")

        if not isinstance(solve, (list, tuple)):
            solve = (solve,) * 2

        if solve[0]:
            fg = solve_fg(fg, alpha)

        fg_, alpha_ = ops.cond(
            ops.random.uniform([], 0.0, 1.0) < prob,
            lambda: compose_batch(
                fg, alpha, trim=trim, solve=solve[1], iterations=iterations
            ),
            lambda: (fg, alpha),
        )

        return fg_, alpha_

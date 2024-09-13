from keras.src import backend
from keras.src import ops

from segme.utils.common.augs.common import apply
from segme.utils.common.augs.common import validate


def shuffle(image, masks, weight, prob, perm=None, name=None):
    with backend.name_scope(name or "shuffle"):
        return apply(
            image,
            masks,
            weight,
            prob,
            lambda x: _shuffle(x, perm),
            None,
            None,
        )


def _shuffle(image, perm=None, name=None):
    with backend.name_scope(name or "shuffle_"):
        image, _, _ = validate(image, None, None)

        if perm is not None:
            perm = backend.convert_to_tensor(perm, "int32")
            if 1 != ops.ndim(perm):
                raise ValueError("Expecting `perm` rank to be 1.")
            image = ops.take(image, perm, axis=-1)
        else:
            image = ops.transpose(image, [3, 0, 1, 2])
            image = ops.random.shuffle(image)
            image = ops.transpose(image, [1, 2, 3, 0])

        return image

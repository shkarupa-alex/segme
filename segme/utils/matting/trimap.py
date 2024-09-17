from keras.src import backend
from keras.src import ops

from segme.utils.common.morph import dilate
from segme.utils.common.morph import erode


def alpha_trimap(alpha, size, name=None):
    with backend.name_scope(name or "alpha_trimap"):
        alpha = backend.convert_to_tensor(alpha, "uint8")

        if 4 != ops.ndim(alpha):
            raise ValueError("Expecting `alpha` rank to be 4.")

        if 1 != alpha.shape[-1]:
            raise ValueError("Expecting `alpha` channels size to be 1.")

        if "uint8" != alpha.dtype:
            raise ValueError("Expecting `alpha` dtype to be `uint8`.")

        if isinstance(size, tuple) and 2 == len(size):
            iterations = ops.random.uniform(
                [2], size[0], size[1] + 1, dtype="int32"
            )
            iterations = ops.unstack(iterations)
        elif isinstance(size, int):
            iterations = size, size
        else:
            raise ValueError(
                "Expecting `size` to be a single margin or "
                "a tuple of [min; max] margins."
            )

        eroded = ops.cast(ops.equal(alpha, 255), "int32")
        eroded = erode(eroded, 3, iterations[0])

        dilated = ops.cast(ops.greater(alpha, 0), "int32")
        dilated = dilate(dilated, 3, iterations[1])

        shape = ops.shape(alpha)
        trimap = ops.full(shape, 128, "uint8")
        trimap = ops.where(ops.equal(eroded, 1 - iterations[0]), 255, trimap)
        trimap = ops.where(ops.equal(dilated, iterations[1]), 0, trimap)

        return trimap

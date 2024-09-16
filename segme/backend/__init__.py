from keras.src.backend.config import backend

if backend() == "torch":
    # When using the torch backend, torch needs to be imported first, otherwise
    # it will segfault upon import.
    import torch

# Import backend functions.
if backend() == "tensorflow":
    from segme.backend.tensorflow import *  # noqa: F403
elif backend() == "jax":
    from segme.backend.jax import *  # noqa: F403
elif backend() == "torch":
    from segme.backend.torch import *  # noqa: F403
elif backend() == "numpy":
    from segme.backend.numpy import *  # noqa: F403
else:
    raise ValueError(f"Unable to import backend : {backend()}")

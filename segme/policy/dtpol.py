import contextlib

from keras.src.dtype_policies import dtype_policy


@contextlib.contextmanager
def policy_scope(policy):
    old_policy = dtype_policy.dtype_policy()
    try:
        dtype_policy.set_dtype_policy(policy)
        yield
    finally:
        dtype_policy.set_dtype_policy(old_policy)

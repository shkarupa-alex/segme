from keras.src.dtype_policies import dtype_policy
from segme.policy import bbpol


def Backbone(scales=None, channels=3, policy=None, dtype=None):
    policy = bbpol.deserialize(policy or bbpol.global_policy())

    prev_dtype = dtype_policy.dtype_policy()
    if dtype is not None:
        dtype_policy.set_dtype_policy(dtype)

    try:
        bbone = bbpol.BACKBONES.new(
            policy.arch_type, policy.init_type, channels, scales
        )
    finally:
        if dtype is not None:
            dtype_policy.set_dtype_policy(prev_dtype)

    return bbone

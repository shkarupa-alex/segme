from segme.policy import bbpol
from segme.policy import dtpol


def Backbone(scales=None, input_tensor=None, policy=None, dtype=None):
    if dtype is not None:
        with dtpol.policy_scope(dtype):
            return Backbone(
                scales=scales,
                input_tensor=input_tensor,
                policy=policy,
                dtype=None,
            )

    policy = bbpol.deserialize(policy or bbpol.global_policy())

    return bbpol.BACKBONES.new(
        policy.arch_type, policy.init_type, scales, input_tensor=input_tensor
    )

from segme.policy import bbpol
from segme.policy import dtpol


def Backbone(scales=None, channels=3, policy=None, dtype=None):
    if dtype is not None:
        with dtpol.policy_scope(dtype):
            return Backbone(
                scales=scales, channels=channels, policy=policy, dtype=None
            )

    policy = bbpol.deserialize(policy or bbpol.global_policy())

    return bbpol.BACKBONES.new(
        policy.arch_type, policy.init_type, channels, scales
    )

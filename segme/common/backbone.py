from segme.policy import bbpol


def Backbone(scales=None, channels=3, policy=None):
    policy = bbpol.deserialize(policy or bbpol.global_policy())

    return bbpol.BACKBONES.new(policy.arch_type, policy.init_type, channels, scales)

from segme.policy import alpol


def Align(filters, scale=2, policy=None, **kwargs):
    policy = alpol.deserialize(policy or alpol.global_policy())

    return alpol.ALIGNERS.new(
        policy.name, filters=filters, scale=scale, **kwargs
    )

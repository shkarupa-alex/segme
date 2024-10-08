import torch


def l2_normalize(x, axis=-1, epsilon=1e-12):
    return torch.nn.functional.normalize(x, p=2.0, dim=axis, eps=epsilon)


def logdet(x):
    raise NotImplementedError


def saturate_cast(x, dtype):
    raise NotImplementedError

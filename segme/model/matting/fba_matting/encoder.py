from tf_keras import models
from segme.policy import bbpol
from segme.policy.backbone.utils import patch_config


def Encoder():
    base_model = bbpol.BACKBONES.new('resnet_rs_50_s8', 'imagenet', 3, [2, 4, 32])
    base_model.trainable = True

    base_config = base_model.get_config()
    base_weights = base_model.get_weights()

    ext_config = patch_config(base_config, [0], 'batch_input_shape', lambda old: old[:-1] + (11,))
    ext_config = patch_config(ext_config, [2], 'mean', lambda old: old + [
        0.306, 0.311, 0.331, 0.402, 0.485, 0.340, 0.417, 0.498])
    ext_config = patch_config(ext_config, [2], 'variance', lambda old: old + [
        0.461 ** 2, 0.463 ** 2, 0.463 ** 2, 0.462 ** 2, 0.450 ** 2, 0.465 ** 2, 0.464 ** 2, 0.452 ** 2])
    ext_model = models.Model.from_config(ext_config)

    ext_weights = []
    for base_weight, ext_weight in zip(base_weights, ext_model.get_weights()):
        if base_weight.shape != ext_weight.shape:
            if base_weight.shape[:2] + base_weight.shape[3:] != ext_weight.shape[:2] + ext_weight.shape[3:]:
                raise ValueError('Unexpected weight shape')

            ext_weight[:, :, :base_weight.shape[2]] = base_weight
            ext_weights.append(ext_weight)
        else:
            ext_weights.append(base_weight)

    ext_model.set_weights(ext_weights)
    ext_model.trainable = True

    return ext_model

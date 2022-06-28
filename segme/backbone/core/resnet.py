from keras import models
from keras.applications import resnet, resnet_v2, resnet_rs
from functools import partial
from ..utils import wrap_bone

ResNet50 = partial(
    wrap_bone,
    resnet.ResNet50,
    resnet.preprocess_input)

ResNet101 = partial(
    wrap_bone,
    resnet.ResNet101,
    resnet.preprocess_input)

ResNet152 = partial(
    wrap_bone,
    resnet.ResNet152,
    resnet.preprocess_input)

ResNet50V2 = partial(
    wrap_bone,
    resnet_v2.ResNet50V2,
    resnet_v2.preprocess_input)

ResNet101V2 = partial(
    wrap_bone,
    resnet_v2.ResNet101V2,
    resnet_v2.preprocess_input)

ResNet152V2 = partial(
    wrap_bone,
    resnet_v2.ResNet152V2,
    resnet_v2.preprocess_input)


def wrap_bone_fix(model, prepr, channels, feats, init, trainable):
    # TODO: wait for https://github.com/keras-team/keras/pull/16726
    base_model = wrap_bone(model, prepr, channels, feats, init, trainable)
    config = base_model.get_config()

    for i, layer in enumerate(config['layers']):
        if 'BatchNormalization' == layer['class_name']:
            config['layers'][i]['config']['momentum'] = 0.99
            config['layers'][i]['config']['epsilon'] = 1.001e-5

    ext_model = models.Model.from_config(config)
    ext_model.set_weights(base_model.get_weights())

    return ext_model


ResNetRS50 = partial(
    wrap_bone_fix,
    resnet_rs.ResNetRS50,
    None)

ResNetRS101 = partial(
    wrap_bone_fix,
    resnet_rs.ResNetRS101,
    None)

ResNetRS152 = partial(
    wrap_bone_fix,
    resnet_rs.ResNetRS152,
    None)

ResNetRS200 = partial(
    wrap_bone_fix,
    resnet_rs.ResNetRS200,
    None)

ResNetRS270 = partial(
    wrap_bone_fix,
    resnet_rs.ResNetRS270,
    None)

ResNetRS350 = partial(
    wrap_bone_fix,
    resnet_rs.ResNetRS350,
    None)

ResNetRS420 = partial(
    wrap_bone_fix,
    resnet_rs.ResNetRS420,
    None)

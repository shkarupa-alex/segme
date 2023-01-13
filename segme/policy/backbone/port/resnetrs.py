import tensorflow as tf
from functools import partial
from keras import backend, layers, models
from keras.applications import imagenet_utils
from keras.applications.resnet_rs import BASE_WEIGHTS_URL, WEIGHT_HASHES, DEPTH_TO_WEIGHT_VARIANTS, BLOCK_ARGS, \
    CONV_KERNEL_INITIALIZER, get_survival_probability, allow_bigger_recursion
from keras.mixed_precision import global_policy
from keras.utils import data_utils, layer_utils
from segme.common.convnormact import Conv, Norm, Act
from segme.policy import cnapol
from segme.policy.backbone.utils import patch_config, wrap_bone
from segme.policy.backbone.backbone import BACKBONES

CONV_KWARGS = {
    'use_bias': False,
    'kernel_initializer': CONV_KERNEL_INITIALIZER
}


def norm_kwargs():
    if 'bn' == cnapol.global_policy().norm_type:
        return {'epsilon': 1e-5}

    return {}


def STEM(name=None):
    if name is None:
        counter = backend.get_uid('stem_')
        name = f'stem_{counter}'

    def apply(inputs):
        x = Conv(32, 3, strides=2, **CONV_KWARGS, name=name + '_stem_conv_1')(inputs)
        x = Norm(**norm_kwargs(), name=name + '_stem_batch_norm_1')(x)
        x = Act(name=name + '_stem_act_1')(x)

        x = Conv(32, 3, **CONV_KWARGS, name=name + '_stem_conv_2')(x)
        x = Norm(**norm_kwargs(), name=name + '_stem_batch_norm_2')(x)
        x = Act(name=name + '_stem_act_2')(x)

        x = Conv(64, 3, **CONV_KWARGS, name=name + '_stem_conv_3')(x)
        x = Norm(**norm_kwargs(), name=name + '_stem_batch_norm_3')(x)
        x = Act(name=name + '_stem_act_3')(x)

        x = Conv(64, 3, strides=2, **CONV_KWARGS, name=name + '_stem_conv_4')(x)
        x = Norm(**norm_kwargs(), name=name + '_stem_batch_norm_4')(x)
        x = Act(name=name + '_stem_act_4')(x)

        return x

    return apply


def SE(in_filters, se_ratio=0.25, expand_ratio=1, name=None):
    if name is None:
        counter = backend.get_uid('se_')
        name = f'se_{counter}'

    def apply(inputs):
        x = layers.GlobalAveragePooling2D(keepdims=True, name=name + '_se_squeeze')(inputs)

        num_reduced_filters = max(1, int(in_filters * 4 * se_ratio))

        x = Conv(num_reduced_filters, 1, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + '_se_reduce')(x)
        x = Act()(x)

        x = layers.Conv2D(
            4 * in_filters * expand_ratio, 1, kernel_initializer=CONV_KERNEL_INITIALIZER, activation='sigmoid',
            name=name + '_se_expand')(x)

        return layers.multiply([inputs, x], name=name + '_se_excite')

    return apply


def BottleneckBlock(filters, strides, use_projection, survival_probability=0.8, name=None):
    if name is None:
        counter = backend.get_uid('block_0_')
        name = f'block_0_{counter}'

    def apply(inputs):
        shortcut = inputs

        if use_projection:
            filters_out = filters * 4
            shortcut = inputs
            if strides == 2:
                shortcut = layers.AveragePooling2D(2, 2, padding='same', name=name + '_projection_pooling', )(shortcut)
            shortcut = Conv(filters_out, 1, **CONV_KWARGS, name=name + '_projection_conv')(shortcut)
            shortcut = Norm(**norm_kwargs(), name=name + '_projection_batch_norm')(shortcut)

        x = Conv(filters, 1, **CONV_KWARGS, name=name + '_conv_1')(inputs)
        x = Norm(**norm_kwargs(), name=name + 'batch_norm_1')(x)
        x = Act(name=name + '_act_1')(x)

        x = Conv(filters, 3, strides=strides, **CONV_KWARGS, name=name + '_conv_2')(x)
        x = Norm(**norm_kwargs(), name=name + '_batch_norm_2')(x)
        x = Act(name=name + '_act_2')(x)

        x = Conv(filters * 4, 1, **CONV_KWARGS, name=name + '_conv_3')(x)
        x = Norm(**norm_kwargs(), name=name + '_batch_norm_3')(x)

        x = SE(filters, name=name + '_se')(x)

        if survival_probability:
            x = layers.Dropout(survival_probability, noise_shape=(None, 1, 1, 1), name=name + '_drop')(x)

        x = layers.add([x, shortcut])

        return Act(name=name + '_output_act')(x)

    return apply


def BlockGroup(filters, strides, num_repeats, survival_probability=0.8, name=None):
    if name is None:
        counter = backend.get_uid('block_group_')
        name = f'block_group_{counter}'

    def apply(inputs):
        x = BottleneckBlock(
            filters=filters, strides=strides, use_projection=True, survival_probability=survival_probability,
            name=name + '_block_0_', )(inputs)

        for i in range(1, num_repeats):
            x = BottleneckBlock(
                filters=filters, strides=1, use_projection=False, survival_probability=survival_probability,
                name=name + f'_block_{i}_', )(x)
        return x

    return apply


def ResNetRS(depth, input_shape=None, dropout_rate=0.25, drop_connect_rate=0.2, include_top=True, block_args=None,
             model_name='resnet-rs', pooling=None, weights='imagenet', input_tensor=None, classes=1000,
             classifier_activation='softmax', include_preprocessing=True):
    available_weight_variants = DEPTH_TO_WEIGHT_VARIANTS[depth]
    if weights == 'imagenet':
        max_input_shape = max(available_weight_variants)
        weights = f'{weights}-i{max_input_shape}'

    weights_allow_list = [f'imagenet-i{x}' for x in available_weight_variants]
    if not (weights in {*weights_allow_list, None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            'The `weights` argument should be either `None` (random initialization), `imagenet` (pre-training on '
            'ImageNet, with highest available input shape), or the path to the weights file to be loaded. '
            f'For ResNetRS{depth} the following weight variants are available {weights_allow_list} (default=highest). '
            f'Received weights={weights}')

    if weights in weights_allow_list and include_top and classes != 1000:
        raise ValueError(
            f'If using `weights` as `imagenet` or any of {weights_allow_list} with `include_top` as true, '
            f'`classes` should be 1000. Received classes={classes}')

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format='channels_last',
        require_flatten=include_top,
        weights=weights,
    )
    input_dtype = global_policy().compute_dtype
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, dtype=input_dtype)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape, dtype=input_dtype)
        else:
            img_input = input_tensor

    x = img_input

    if include_preprocessing:
        num_channels = input_shape[-1]
        x = layers.Rescaling(scale=1.0 / 255, name='rescale')(x)
        if num_channels == 3:
            x = layers.Normalization(
                mean=[0.485, 0.456, 0.406], variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2], name='normalize')(x)

    x = STEM(name='stem_1')(x)

    if block_args is None:
        block_args = BLOCK_ARGS[depth]

    for i, args in enumerate(block_args):
        survival_probability = get_survival_probability(
            init_rate=drop_connect_rate, block_num=i + 2, total_blocks=len(block_args) + 1)

        x = BlockGroup(
            filters=args['input_filters'], strides=(1 if i == 0 else 2), num_repeats=args['num_repeats'],
            survival_probability=survival_probability, name=f'BlockGroup{i + 2}_')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)

        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(
            classes, activation=classifier_activation, name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = models.Model(inputs, x, name=model_name)

    if weights in weights_allow_list:
        weights_input_shape = weights.split('-')[-1]
        weights_name = f'{model_name}-{weights_input_shape}'
        if not include_top:
            weights_name += '_notop'

        filename = f'{weights_name}.h5'
        download_url = BASE_WEIGHTS_URL + filename
        weights_path = data_utils.get_file(
            fname=filename,
            origin=download_url,
            cache_subdir='models',
            file_hash=WEIGHT_HASHES[filename],
        )
        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model


def ResNetRS50(include_top=True, weights='imagenet', classes=1000, input_shape=None, input_tensor=None, pooling=None,
               classifier_activation='softmax', include_preprocessing=True, ):
    return ResNetRS(
        depth=50, include_top=include_top, drop_connect_rate=0.0, dropout_rate=0.25, weights=weights, classes=classes,
        input_shape=input_shape, input_tensor=input_tensor, pooling=pooling,
        classifier_activation=classifier_activation, model_name='resnet-rs-50',
        include_preprocessing=include_preprocessing, )


def ResNetRS101(include_top=True, weights='imagenet', classes=1000, input_shape=None, input_tensor=None, pooling=None,
                classifier_activation='softmax', include_preprocessing=True, ):
    return ResNetRS(
        depth=101, include_top=include_top, drop_connect_rate=0.0, dropout_rate=0.25, weights=weights, classes=classes,
        input_shape=input_shape, input_tensor=input_tensor, pooling=pooling,
        classifier_activation=classifier_activation, model_name='resnet-rs-101',
        include_preprocessing=include_preprocessing, )


def ResNetRS152(include_top=True, weights='imagenet', classes=1000, input_shape=None, input_tensor=None, pooling=None,
                classifier_activation='softmax', include_preprocessing=True, ):
    return ResNetRS(
        depth=152, include_top=include_top, drop_connect_rate=0.0, dropout_rate=0.25, weights=weights, classes=classes,
        input_shape=input_shape, input_tensor=input_tensor, pooling=pooling,
        classifier_activation=classifier_activation, model_name='resnet-rs-152',
        include_preprocessing=include_preprocessing, )


def ResNetRS200(include_top=True, weights='imagenet', classes=1000, input_shape=None, input_tensor=None, pooling=None,
                classifier_activation='softmax', include_preprocessing=True, ):
    return ResNetRS(
        depth=200, include_top=include_top, drop_connect_rate=0.1, dropout_rate=0.25, weights=weights, classes=classes,
        input_shape=input_shape, input_tensor=input_tensor, pooling=pooling,
        classifier_activation=classifier_activation, model_name='resnet-rs-200',
        include_preprocessing=include_preprocessing, )


def ResNetRS270(include_top=True, weights='imagenet', classes=1000, input_shape=None, input_tensor=None, pooling=None,
                classifier_activation='softmax', include_preprocessing=True, ):
    allow_bigger_recursion(1300)
    return ResNetRS(
        depth=270, include_top=include_top, drop_connect_rate=0.1, dropout_rate=0.25, weights=weights, classes=classes,
        input_shape=input_shape, input_tensor=input_tensor, pooling=pooling,
        classifier_activation=classifier_activation, model_name='resnet-rs-270',
        include_preprocessing=include_preprocessing, )


def ResNetRS350(include_top=True, weights='imagenet', classes=1000, input_shape=None, input_tensor=None, pooling=None,
                classifier_activation='softmax', include_preprocessing=True, ):
    allow_bigger_recursion(1500)
    return ResNetRS(
        depth=350, include_top=include_top, drop_connect_rate=0.1, dropout_rate=0.4, weights=weights, classes=classes,
        input_shape=input_shape, input_tensor=input_tensor, pooling=pooling,
        classifier_activation=classifier_activation, model_name='resnet-rs-350',
        include_preprocessing=include_preprocessing, )


def ResNetRS420(include_top=True, weights='imagenet', classes=1000, input_shape=None, input_tensor=None, pooling=None,
                classifier_activation='softmax', include_preprocessing=True):
    allow_bigger_recursion(1800)
    return ResNetRS(
        depth=420, include_top=include_top, dropout_rate=0.4, drop_connect_rate=0.1, weights=weights, classes=classes,
        input_shape=input_shape, input_tensor=input_tensor, pooling=pooling,
        classifier_activation=classifier_activation, model_name='resnet-rs-420',
        include_preprocessing=include_preprocessing)


def wrap_bone_policy(model, prepr, init, channels, end_points, name):
    if 'conv-bn-relu' == cnapol.global_policy().name or init is None:
        return wrap_bone(model, prepr, init, channels, end_points, name)

    with cnapol.policy_scope('conv-bn-relu'):
        base_model = wrap_bone(model, prepr, init, channels, end_points, name)

    base_weights = {w.name: w for w in base_model.weights}
    if len(base_model.weights) != len(base_weights.keys()):
        raise ValueError('Some weights have equal names')

    ext_model = wrap_bone(model, prepr, None, channels, end_points, name)

    ext_weights = []
    for random_weight in ext_model.weights:
        weight_name = random_weight.name

        if 'batch_norm' in weight_name:
            ext_weights.append(random_weight)
            continue

        if weight_name not in base_weights:
            ext_weights.append(random_weight)
            continue

        if random_weight.shape != base_weights[weight_name].shape:
            ext_weights.append(random_weight)
            continue

        ext_weights.append(base_weights[weight_name])

    ext_model.set_weights(ext_weights)
    ext_model.trainable = True

    return ext_model


BACKBONES.register('resnet_rs_50')((
    partial(wrap_bone_policy, ResNetRS50, None), [
        None, 'stem_1_stem_act_3', 'BlockGroup2__block_2__output_act', 'BlockGroup3__block_3__output_act',
        'BlockGroup4__block_5__output_act', 'BlockGroup5__block_2__output_act']))

BACKBONES.register('resnet_rs_101')((
    partial(wrap_bone_policy, ResNetRS101, None), [
        None, 'stem_1_stem_act_3', 'BlockGroup2__block_2__output_act', 'BlockGroup3__block_3__output_act',
        'BlockGroup4__block_22__output_act', 'BlockGroup5__block_2__output_act']))

BACKBONES.register('resnet_rs_152')((
    partial(wrap_bone_policy, ResNetRS152, None), [
        None, 'stem_1_stem_act_3', 'BlockGroup2__block_2__output_act', 'BlockGroup3__block_7__output_act',
        'BlockGroup4__block_35__output_act', 'BlockGroup5__block_2__output_act']))

BACKBONES.register('resnet_rs_200')((
    partial(wrap_bone_policy, ResNetRS200, None), [
        None, 'stem_1_stem_act_3', 'BlockGroup2__block_2__output_act', 'BlockGroup3__block_23__output_act',
        'BlockGroup4__block_35__output_act', 'BlockGroup5__block_2__output_act']))

BACKBONES.register('resnet_rs_270')((
    partial(wrap_bone_policy, ResNetRS270, None), [
        None, 'stem_1_stem_act_3', 'BlockGroup2__block_3__output_act', 'BlockGroup3__block_28__output_act',
        'BlockGroup4__block_52__output_act', 'BlockGroup5__block_3__output_act']))

BACKBONES.register('resnet_rs_350')((
    partial(wrap_bone_policy, ResNetRS350, None), [
        None, 'stem_1_stem_act_3', 'BlockGroup2__block_3__output_act', 'BlockGroup3__block_35__output_act',
        'BlockGroup4__block_71__output_act', 'BlockGroup5__block_3__output_act']))

BACKBONES.register('resnet_rs_420')((
    partial(wrap_bone_policy, ResNetRS420, None), [
        None, 'stem_1_stem_act_3', 'BlockGroup2__block_3__output_act', 'BlockGroup3__block_43__output_act',
        'BlockGroup4__block_86__output_act', 'BlockGroup5__block_3__output_act']))


def wrap_bone_stride8(model, prepr, init, channels, end_points, name):
    base_model = wrap_bone_policy(model, prepr, init, channels, end_points, name)
    ext_config = base_model.get_config()

    stride_patches = []
    for layer in ext_config['layers']:
        if 'SegMe>Common>ConvNormAct>Conv' != layer['class_name']:
            continue
        if layer['config']['kernel_size'] in {1, (1, 1)}:
            continue

        if layer['config']['name'].startswith('BlockGroup4__'):
            dilation = 2
        elif layer['config']['name'].startswith('BlockGroup5__'):
            dilation = 4
        else:
            continue

        if layer['config']['strides'] not in {1, (1, 1)}:
            assert layer['config']['strides'] in {2, (2, 2)}
            dilation = dilation // 2

        stride_patches.append((layer['config']['name'], dilation))

    for layer, dilation in stride_patches:
        ext_config = patch_config(ext_config, [layer], 'strides', 1)
        ext_config = patch_config(ext_config, [layer], 'dilation_rate', dilation)

    for i, layer in enumerate(ext_config['layers']):
        if layer['config']['name'] not in {
            'BlockGroup4__block_0__projection_pooling', 'BlockGroup5__block_0__projection_pooling'}:
            continue

        layer['class_name'] = 'Activation'
        layer['config']['activation'] = 'linear'
        del layer['config']['pool_size']
        del layer['config']['padding']
        del layer['config']['strides']
        del layer['config']['data_format']

        ext_config['layers'][i] = layer

    ext_model = models.Model.from_config(ext_config)
    ext_model.set_weights(base_model.get_weights())
    ext_model.trainable = True

    return ext_model


BACKBONES.register('resnet_rs_50_s8')((
    partial(wrap_bone_stride8, ResNetRS50, None), [
        None, 'stem_1_stem_act_3', 'BlockGroup2__block_2__output_act', 'BlockGroup3__block_3__output_act',
        'BlockGroup4__block_5__output_act', 'BlockGroup5__block_2__output_act']))

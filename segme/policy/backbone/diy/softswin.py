import numpy as np
import tensorflow as tf
from functools import partial
from keras import backend, initializers, layers, models
from keras.mixed_precision import global_policy
from keras.src.applications import imagenet_utils
from keras.src.applications.efficientnet_v2 import CONV_KERNEL_INITIALIZER
from keras.src.utils import data_utils, layer_utils
from segme.common.convnormact import Norm, Conv, Act
from segme.common.drop import SlicePath, RestorePath
from segme.common.attn import SwinAttention
from segme.model.classification.data import tree_class_map
from segme.policy import cnapol
from segme.policy.backbone.utils import wrap_bone
from segme.policy.backbone.backbone import BACKBONES

BASE_URL = 'https://github.com/shkarupa-alex/segme/releases/download/{}.h5'
WEIGHT_URLS = {
    'soft_swin_tiny_conv-ln-gelu': BASE_URL.format('2.2.1/soft_swin_tiny_h21k'),
    'soft_swin_small_conv-ln-gelu': BASE_URL.format('2.2.1/soft_swin_small_h21k'),
    'soft_swin_base_conv-ln-gelu': BASE_URL.format('2.2.1/soft_swin_base_h21k')
}
WEIGHT_HASHES = {
    'soft_swin_tiny_conv-ln-gelu': 'a99bb80498ab5b531c27e904bfbabd6fbb979e1ff1d07874113c4b6b40c1d55a',
    'soft_swin_small_conv-ln-gelu': '74b6e609b49d013f32c8a4f12b845cd9b3d0782e9a3b4c4e4d1b9df18a0a0129',
    'soft_swin_base_conv-ln-gelu': '7807ba88d944f7b1d59a47557dfd83a68fda48cc9116ae23e804fbe7cb4c4fd1'
}


def Stem(filters, depth, path_gamma=1., path_drop=0., name=None):
    if name is None:
        counter = backend.get_uid('stem')
        name = f'stem_{counter}'

    if isinstance(path_gamma, float):
        path_gamma = [path_gamma] * depth
    else:
        if len(path_gamma) != depth:
            raise ValueError('Number of path gammas must equals to depth.')

    if isinstance(path_drop, float):
        path_drop = [path_drop] * depth
    else:
        if len(path_drop) != depth:
            raise ValueError('Number of path dropouts must equals to depth.')

    def apply(inputs):
        x = Conv(filters, 3, strides=2, kernel_initializer=CONV_KERNEL_INITIALIZER, name=f'{name}_0_conv')(inputs)
        x = Act(name=f'{name}_0_act')(x)
        x = Norm(center=False, name=f'{name}_0_norm')(x)

        for i in range(depth):
            y, indices = SlicePath(path_drop[i], name=f'{name}_{i + 1}_slice')(x)
            y = Conv(filters, 3, kernel_initializer=CONV_KERNEL_INITIALIZER, name=f'{name}_{i + 1}_conv')(y)
            y = Act(name=f'{name}_{i + 1}_act')(y)
            y = Norm(
                center=False, gamma_initializer=initializers.Constant(path_gamma[i]), name=f'{name}_{i + 1}_norm')(y)
            y = RestorePath(path_drop[i], name=f'{name}_{i + 1}_drop')([y, indices])
            x = layers.add([y, x], name=f'{name}_{i + 1}_add')

        return x

    return apply


def Reduce(fused, kernel_size=3, expand_ratio=2., project_ratio=2., name=None):
    if name is None:
        counter = backend.get_uid('reduce')
        name = f'reduce_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        expand_filters = round(channels * expand_ratio)
        project_filters = round(channels * project_ratio)

        if fused:  # From EfficientNet2
            x = Conv(
                expand_filters, kernel_size, strides=2, kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=f'{name}_expand')(inputs)
        else:
            x = Conv(expand_filters, 1, use_bias=False, name=f'{name}_expand_pw')(inputs)
            x = Conv(
                None, kernel_size, strides=2, kernel_initializer=CONV_KERNEL_INITIALIZER, name=f'{name}_expand_dw')(x)
        x = Act(name=f'{name}_act')(x)
        x = Conv(project_filters, 1, use_bias=False, name=f'{name}_squeeze')(x)
        x = Norm(center=False, name=f'{name}_norm')(x)

        return x

    return apply


def MLP(expand_ratio=4., path_drop=0., gamma_initializer='ones', name=None):
    if name is None:
        counter = backend.get_uid('mlp')
        name = f'mlp_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        expand_filters = int(channels * expand_ratio)

        x, indices = SlicePath(path_drop, name=f'{name}_slice')(inputs)
        x = layers.Dense(expand_filters, name=f'{name}_expand')(x)
        x = Act(name=f'{name}_act')(x)
        x = layers.Dense(channels, name=f'{name}_squeeze')(x)
        x = Norm(gamma_initializer=gamma_initializer, name=f'{name}_norm')(x)
        x = RestorePath(path_drop, name=f'{name}_drop')([x, indices])
        x = layers.add([x, inputs], name=f'{name}_add')

        return x

    return apply


def AttnBlock(current_window, pretrain_window, num_heads, shift_mode, path_drop=0., expand_ratio=4.,
              path_gamma=1., name=None):
    if name is None:
        counter = backend.get_uid('attn_block')
        name = f'attn_block_{counter}'

    gamma_initializer = initializers.Constant(path_gamma)

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        x, indices = SlicePath(path_drop, name=f'{name}_slice')(inputs)
        x = SwinAttention(
            current_window, pretrain_window, num_heads, shift_mode=shift_mode, name=f'{name}_swin_attn')(x)
        x = Norm(gamma_initializer=gamma_initializer, name=f'{name}_swin_norm')(x)
        x = RestorePath(path_drop, name=f'{name}_drop')([x, indices])
        x = layers.add([x, inputs], name=f'{name}_swin_add')

        x = MLP(
            expand_ratio=expand_ratio, path_drop=path_drop, gamma_initializer=gamma_initializer, name=f'{name}_mlp')(x)

        return x

    return apply


def SoftSwin(
        embed_dim, stem_depth, stage_depths, pretrain_window, current_window=None, expand_ratio=4, path_gamma=0.01,
        path_drop=0.2, pretrain_size=384, current_size=None, input_shape=None, include_top=True, model_name='soft_swin',
        pooling=None, weights=None, input_tensor=None, classes=1000, classifier_activation='softmax',
        include_preprocessing=False):
    if embed_dim % 32:
        raise ValueError('Embedding size should be a multiple of 32.')

    if len(stage_depths) < 4:
        raise ValueError('Number of stages should be greater then 4.')

    current_window = current_window or pretrain_window

    if weights not in {'imagenet', None} and not tf.io.gfile.exists(weights):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), or the path to the weights file to be loaded.')

    tree_classes = len(set(tree_class_map().values()))
    if weights == 'imagenet' and include_top and classes not in {1000, tree_classes}:
        raise ValueError(f'If using `weights` as `"imagenet"` with `include_top` as true, `classes` should be '
                         f'1000 or {tree_classes} depending on pretrain dataset.')

    if input_tensor is not None:
        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(f'Expecting `input_tensor` to be a symbolic tensor instance. '
                             f'Got {input_tensor} of type {type(input_tensor)}')

    if input_tensor is not None:
        tensor_shape = backend.int_shape(input_tensor)[1:]
        if input_shape and tensor_shape != input_shape:
            raise ValueError('Shape of `input_tensor` should equals to `input_shape` if both provided.')
        else:
            input_shape = tensor_shape

    # Determine proper input shape
    min_size = 2 ** (len(stage_depths) + 1)
    pretrain_size = pretrain_size or pretrain_window * min_size
    current_size = current_size or pretrain_size
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape, default_size=current_size, min_size=min_size, data_format='channel_last', require_flatten=False,
        weights=weights)
    input_dtype = global_policy().compute_dtype

    if input_tensor is not None:
        if backend.is_keras_tensor(input_tensor):
            image = input_tensor
        else:
            image = layers.Input(shape=input_shape, name='images', dtype=input_dtype, tensor=input_tensor)
    else:
        image = layers.Input(shape=input_shape, name='images', dtype=input_dtype)

    x = image

    if include_preprocessing:
        x = layers.Rescaling(scale=1.0 / 255, name='rescale')(x)
        x = layers.Normalization(
            mean=[0.485, 0.456, 0.406], variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2], name='normalize')(x)

    path_gammas = np.linspace(path_gamma, 1e-5, stem_depth + sum(stage_depths)).tolist()
    stem_gammas, path_gammas = path_gammas[:stem_depth], path_gammas[stem_depth:]

    path_drops = np.linspace(0., path_drop, stem_depth + sum(stage_depths)).tolist()
    stem_drops, path_drops = path_drops[:stem_depth], path_drops[stem_depth:]

    stem_filters = np.ceil(embed_dim / 4 / 8).astype('int32').item() * 8
    x = Stem(stem_filters, stem_depth, path_gamma=stem_gammas, path_drop=stem_drops, name='stem')(x)
    x = layers.Activation('linear', name='stem_out')(x)

    shift_counter = -1
    for i, stage_depth in enumerate(stage_depths):
        stage_window = min(current_window, current_size // 2 ** (i + 2))
        num_heads = embed_dim // 2 ** (5 - i)

        stage_gammas, path_gammas = path_gammas[:stage_depth], path_gammas[stage_depth:]
        stage_drops, path_drops = path_drops[:stage_depth], path_drops[stage_depth:]

        reduce_fused = 0 == i  # From EfficientNet2
        reduce_ratio = embed_dim / stem_filters if 0 == i else 2
        x = Reduce(
            reduce_fused, expand_ratio=reduce_ratio, project_ratio=reduce_ratio, name=f'stage_{i}_reduce')(x)

        for j in range(stage_depth):
            shift_counter += j % 2
            shift_mode = shift_counter % 4 + 1 if j % 2 else 0
            x = AttnBlock(
                stage_window, pretrain_window, num_heads, shift_mode, expand_ratio=expand_ratio,
                path_gamma=stage_gammas[j], path_drop=stage_drops[j], name=f'stage_{i}_attn_{j}')(x)

        x = layers.Activation('linear', name=f'stage_{i}_out')(x)

    x = Norm(name='norm')(x)

    if include_top or pooling in {None, 'avg'}:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    else:
        raise ValueError(f'Expecting pooling to be one of None/avg/max. Found: {pooling}')

    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, name='head')(x)
    x = layers.Activation(classifier_activation, dtype='float32', name='pred')(x)

    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = image

    model = models.Model(inputs, x, name=model_name)

    weights_key = f'{model_name}_{cnapol.global_policy().name}'
    if 'imagenet' == weights and weights_key in WEIGHT_URLS:
        weights_url = WEIGHT_URLS[weights_key]
        weights_hash = WEIGHT_HASHES[weights_key]
        weights_path = data_utils.get_file(origin=weights_url, file_hash=weights_hash, cache_subdir='soft_swin')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if include_top:
        return model

    outputs = model.get_layer(name='norm').output
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def SoftSwinTiny(
        embed_dim=96, stem_depth=2, stage_depths=(2, 2, 6, 2), pretrain_window=16, pretrain_size=256,
        model_name='soft_swin_tiny', weights='imagenet', classes=14607, classifier_activation='linear', **kwargs):
    # 28.0 14.3
    # 0.1993 0.6269 0.7064
    with cnapol.policy_scope('conv-ln-gelu'):
        return SoftSwin(
            embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, pretrain_window=pretrain_window,
            pretrain_size=pretrain_size, model_name=model_name, weights=weights, classes=classes,
            classifier_activation=classifier_activation, **kwargs)


def SoftSwinSmall(
        embed_dim=96, stem_depth=2, stage_depths=(2, 2, 18, 2), pretrain_window=16, path_drop=0.3, pretrain_size=256,
        model_name='soft_swin_small', weights='imagenet', classes=14607, **kwargs):
    # 49.4 26.7
    # 0.1993 0.6527 0.7229
    with cnapol.policy_scope('conv-ln-gelu'):
        return SoftSwin(
            embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, pretrain_window=pretrain_window,
            path_drop=path_drop, pretrain_size=pretrain_size, model_name=model_name, weights=weights, classes=classes,
            **kwargs)


def SoftSwinBase(
        embed_dim=128, stem_depth=3, stage_depths=(2, 2, 18, 2), current_window=24, pretrain_window=12,
        pretrain_size=192, current_size=384, model_name='soft_swin_base', weights=None, classes=14607, **kwargs):
    # 87.3 50.6 @ 256
    # 0.1994 0.6589 0.7476
    with cnapol.policy_scope('conv-ln-gelu'):
        return SoftSwin(
            embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, current_window=current_window,
            pretrain_window=pretrain_window, pretrain_size=pretrain_size, current_size=current_size,
            model_name=model_name, weights=weights, classes=classes, **kwargs)


def SoftSwinLarge(
        embed_dim=192, stem_depth=4, stage_depths=(2, 2, 18, 2), current_window=24, pretrain_window=12,
        pretrain_size=192, current_size=384, model_name='soft_swin_large', weights=None, classes=14607, **kwargs):
    # 195.4 107.4 @ 256
    with cnapol.policy_scope('conv-ln-gelu'):
        return SoftSwin(
            embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, current_window=current_window,
            pretrain_window=pretrain_window, pretrain_size=pretrain_size, current_size=current_size,
            model_name=model_name, weights=weights, classes=classes, **kwargs)


BACKBONES.register('softswin_tiny')((
    partial(wrap_bone, SoftSwinTiny, None), [
        None, 'stem_out', 'stage_0_out', 'stage_1_out', 'stage_2_out', 'stage_3_out']))

BACKBONES.register('softswin_small')((
    partial(wrap_bone, SoftSwinSmall, None), [
        None, 'stem_out', 'stage_0_out', 'stage_1_out', 'stage_2_out', 'stage_3_out']))

BACKBONES.register('softswin_base')((
    partial(wrap_bone, SoftSwinBase, None), [
        None, 'stem_out', 'stage_0_out', 'stage_1_out', 'stage_2_out', 'stage_3_out']))

BACKBONES.register('softswin_large')((
    partial(wrap_bone, SoftSwinLarge, None), [
        None, 'stem_out', 'stage_0_out', 'stage_1_out', 'stage_2_out', 'stage_3_out']))

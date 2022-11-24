import numpy as np
import tensorflow as tf
from keras import backend, initializers, layers, models
from keras.applications import imagenet_utils
from keras.applications.efficientnet_v2 import CONV_KERNEL_INITIALIZER
from keras.mixed_precision import global_policy
from keras.utils import data_utils, layer_utils
from segme.common.convnormact import Norm, ConvNormAct, ConvNorm
from segme.common.drop import DropPath
from segme.common.mbconv import MBConv
from segme.policy.backbone.diy.coma.attn import DHMSA, GGMSA, CHMSA
from segme.policy.backbone.diy.coma.mlp import MLP

WEIGHT_URLS = {}
WEIGHT_HASHES = {}


# TODO: DW in MLP - before/after/none
# TODO: SE in MLP - yes/no
# TODO: DW in attention - yes/no

# TODO: disable biases?
# TODO: weight regularization in Dense?
# TODO: channel shift

# TODO: CLS token usage https://github.com/microsoft/CvT/blob/main/lib/models/cls_cvt.py#L183
# TODO: IN21k pretraining https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/98f13708210194c475687be6106a3b84-Paper-round1.pdf


def Stem(filters, depth, path_gamma=1., name=None):
    if name is None:
        counter = backend.get_uid('stem')
        name = f'stem_{counter}'

    def apply(x):
        x = ConvNormAct(filters, 3, strides=2, kernel_initializer=CONV_KERNEL_INITIALIZER, name=f'{name}_conv')(x)

        # From EfficientNet2
        for i in range(depth):
            x = MBConv(
                filters, 3, fused=True, expand_ratio=1., se_ratio=0.,
                gamma_initializer=initializers.constant(path_gamma), name=f'{name}_mbconv_{i}')(x)

        return x

    return apply


def Reduce(fused, kernel_size=3, spatial_ratio=0.5, channel_ratio=2., expand_ratio=4., se_ratio=0.25, drop_ratio=0.,
           path_gamma=1., name=None):
    if name is None:
        counter = backend.get_uid('reduce')
        name = f'reduce_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        strides = int(1. / spatial_ratio)
        filters = int(channels * channel_ratio)

        # From ResNetRS
        skip = layers.AvgPool2D(strides, name=f'{name}_pool')(inputs)
        skip = ConvNorm(filters, 1, name=f'{name}_proj')(skip)

        # From EfficientNet2
        x = MBConv(
            filters, kernel_size, fused=fused, strides=strides, expand_ratio=expand_ratio, se_ratio=se_ratio,
            gamma_initializer=initializers.constant(path_gamma), name=f'{name}_mbconv')(inputs)
        x = DropPath(drop_ratio, name=f'{name}_drop')(x)

        x = layers.add([x, skip], name=f'{name}_add')

        return x

    return apply


def ConvBlock(fused, kernel_size=3, expand_ratio=4., se_ratio=0.25, drop_ratio=0., path_gamma=1., name=None):
    if name is None:
        counter = backend.get_uid('conv_block')
        name = f'conv_block_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        x = MBConv(
            channels, kernel_size, fused, expand_ratio=expand_ratio, se_ratio=se_ratio,
            gamma_initializer=initializers.constant(path_gamma), drop_ratio=drop_ratio, name=f'{name}_mbconv')(inputs)

        return x

    return apply


def WindBlock(current_window, pretrain_window, num_heads, dilation_rate=1, qkv_bias=True, attn_drop=0., proj_drop=0.,
              path_drop=0., mlp_ratio=4., mlp_drop=0., path_gamma=1., name=None):
    if name is None:
        counter = backend.get_uid('attn_block')
        name = f'attn_block_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        x = DHMSA(
            current_window, pretrain_window, num_heads, dilation_rate=dilation_rate, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=proj_drop, name=f'{name}_window')(inputs)
        x = Norm(gamma_initializer=initializers.constant(path_gamma), name=f'{name}_norm1')(x)
        x = DropPath(path_drop, name=f'{name}_drop1')(x)
        x = layers.add([x, inputs], name=f'{name}_add1')

        y = MLP(mlp_ratio, mlp_drop, name=f'{name}_mlp')(x)
        y = Norm(gamma_initializer=initializers.constant(path_gamma), name=f'{name}_norm2')(y)
        y = DropPath(path_drop, name=f'{name}_drop2')(y)
        y = layers.add([y, x], name=f'{name}_add2')

        return y

    return apply


def GridBlock(current_window, pretrain_window, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., path_drop=0.,
              mlp_ratio=4., mlp_drop=0., path_gamma=1., name=None):
    if name is None:
        counter = backend.get_uid('attn_block')
        name = f'attn_block_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        x = GGMSA(
            current_window, pretrain_window, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,
            name=f'{name}_grid')(inputs)
        x = Norm(gamma_initializer=initializers.constant(path_gamma), name=f'{name}_norm1')(x)
        x = DropPath(path_drop, name=f'{name}_drop1')(x)
        x = layers.add([x, inputs], name=f'{name}_add1')

        y = MLP(mlp_ratio, mlp_drop, name=f'{name}_mlp')(x)
        y = Norm(gamma_initializer=initializers.constant(path_gamma), name=f'{name}_norm2')(y)
        y = DropPath(path_drop, name=f'{name}_drop2')(y)
        y = layers.add([y, x], name=f'{name}_add2')

        return y

    return apply


def ChanBlock(num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., path_drop=0., mlp_ratio=4., mlp_drop=0.,
              path_gamma=1., name=None):
    if name is None:
        counter = backend.get_uid('attn_block')
        name = f'attn_block_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        x = CHMSA(
            num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, name=f'{name}_channel')(inputs)
        x = Norm(gamma_initializer=initializers.constant(path_gamma), name=f'{name}_norm1')(x)
        x = DropPath(path_drop, name=f'{name}_drop1')(x)
        x = layers.add([x, inputs], name=f'{name}_add1')

        y = MLP(mlp_ratio, mlp_drop, name=f'{name}_mlp')(x)
        y = Norm(gamma_initializer=initializers.constant(path_gamma), name=f'{name}_norm2')(y)
        y = DropPath(path_drop, name=f'{name}_drop2')(y)
        y = layers.add([y, x], name=f'{name}_add2')

        return y

    return apply


def CoMA(
        embed_dim, stem_depth, stage_depths, current_window=8, pretrain_window=8, path_gamma=0.1, path_drop=0.2,
        pretrain_size=384, input_shape=None, include_top=True, model_name='coma', pooling=None, weights='imagenet',
        input_tensor=None, classes=21841, classifier_activation='softmax', include_preprocessing=True):
    """ Inspired with:

    15.11.2022 Focal Modulation Networks
        + overlapped patch embedding
        ~ deeper but thinner
        - focal-modulation instead of self-attention
        - context aggregation
    10.11.2022 Demystify Transformers & Convolutions in Modern Image Deep Networks
        + overlapped patch embedding and reduction
        + haloing for local-attention spatial token mixer
    24.10.2022 MetaFormer Baselines for Vision
        + stage architecrure CCTT
        ~ disable all biases
        - scaling the residual branch
        - convolutional block with separated spatial mixer & MLP
        - StarReLU with learnable scale and bias
    01.10.2022 Global Context Vision Transformers
        + stride-2 stem
        ~ modified Fused-MBConv block for reduction
        ~ stage ratio 3:4:19:5
        - global query generation
    29.09.2022 Dilated Neighborhood Attention Transformer
        + dilated (sparse) window self-attention
        + gradual dilation order (1, 2, 1, 4, 1, 6)
    09.09.2022 MaxViT: Multi-Axis Vision Transformer
        + grid self-attention
        ~ stage ratio 1:1:9:1
    16.05.2022 Activating More Pixels in Image Super-Resolution Transformer
        + overlapping window cross-attention
        ~ channel attention
        ~ enlarging window size of self-attention
    11.04.2022 Swin Transformer V2: Scaling Up Capacity and Resolution
        + log-spaced continuous position bias
        + residual post normalization
        + scaled cosine attention
    07.04.2022 DaViT: Dual Attention Vision Transformers
        + channel group self-attention
    02.03.2022 A ConvNet for the 2020s
        + stage ratio 1:1:9:1
        + adding normalization layers wherever spatial resolution is changed
    28.10.2021 SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
        + overlapped reduction
        + depthwise convolution in MLP
        - reduce length for efficient self-attention
    15.09.2021 CoAtNet: Marrying Convolution and Attention for All Data Sizes
        + stage architecrure CCTT
        + MBConv for reduction and as convolutional block
        + stride-2 stem
        + stage ratio 1:3:7:1
    07.06.2021 Scaling Local Self-Attention for Parameter Efficient Visual Backbones
        + overlapping window with halo = 1/2 of window size
        + stage architecrure CCTT
        + accuracy consistently improves as the window size increases
    29.03.2021 CvT: Introducing Convolutions to Vision Transformers
        + overlapped patch embedding and reduction
        + depthwise convolution in attention projection
        - query & key projection with stride 2
    13.03.2021 Revisiting ResNets: Improved Training and Scaling Strategies
        + average pooling & pointwise convolution as main reduction branch
        ~ zero-gamma trick
        - decreasing weight decay when using augmentations

    """
    if embed_dim % 32:
        raise ValueError('Embedding size should be a multiple of 32.')

    if len(stage_depths) < 4:
        raise ValueError('Number of stages should be greater then 4.')

    if weights not in {'imagenet', None} and not tf.io.gfile.exists(weights):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes not in {1000, 21841}:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` as true, '
                         '`classes` should be 1000 or 21841 depending on pretrain dataset.')

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
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape, default_size=pretrain_size, min_size=min_size, data_format='channel_last', require_flatten=False,
        weights=weights)

    if input_tensor is not None:
        if backend.is_keras_tensor(input_tensor):
            image = input_tensor
        else:
            image = layers.Input(tensor=input_tensor, shape=input_shape, dtype=global_policy().compute_dtype)
    else:
        image = layers.Input(shape=input_shape)

    x = image

    if include_preprocessing:
        x = layers.Rescaling(scale=1.0 / 255, name='rescale')(x)
        if 3 == input_shape[-1]:
            x = layers.Normalization(
                mean=[0.485, 0.456, 0.406], variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2], name='normalize')(x)

    path_drops = np.linspace(0., path_drop, sum(stage_depths) + len(stage_depths))
    path_gammas = np.linspace(path_gamma, 1e-4, sum(stage_depths) + len(stage_depths) * 2 + 1)

    x = Stem(embed_dim // 2, stem_depth, path_gamma=path_gammas[0], name='stem')(x)
    x = layers.Activation('linear', name='stem_out')(x)

    for i, depth in enumerate(stage_depths):
        fused = embed_dim * (2 ** i) < 128
        num_heads = embed_dim // 2 ** (5 - i)
        path_drop_ = path_drops[sum(stage_depths[:i]) + i:sum(stage_depths[:i + 1]) + i + 1].tolist()
        path_gamma_ = path_gammas[sum(stage_depths[:i]) + i * 2 + 1:sum(stage_depths[:i + 1]) + i * 2 + 3].tolist()

        # From GCViT
        x = Reduce(fused, path_gamma=path_gamma_[0], name=f'stage_{i}_reduce')(x)

        for j in range(depth):
            if i < 2:  # From CoAtNet
                x = ConvBlock(
                    fused, drop_ratio=path_drop_[j], path_gamma=path_gamma_[j + 1], name=f'stage_{i}_conv_{j}')(x)
                continue

            if len(stage_depths) - 1 == i and j % 2:
                x = GridBlock(  # From MaxViT
                    current_window, pretrain_window, num_heads, path_gamma=path_gamma_[j + 1], path_drop=path_drop_[j],
                    name=f'stage_{i}_attn_{j}')(x)
                continue

            current_size = pretrain_size // 2 ** (i + 2)
            dilation_max = current_size * 2 // (3 * current_window)
            dilation_rate = 1 + (j % 2) * (j // 2 % max(1, dilation_max - 1) + 1)
            dilation_rate = min(dilation_rate, dilation_max)
            dilation_rate = max(dilation_rate, 1)

            x = WindBlock(
                current_window, pretrain_window, num_heads, dilation_rate=dilation_rate, path_gamma=path_gamma_[j + 1],
                path_drop=path_drop_[j], name=f'stage_{i}_attn_{j}')(x)

        # From DaViT, HAT
        x = ChanBlock(
            num_heads, path_gamma=path_gamma_[-1], path_drop=path_drop_[-1], name=f'stage_{i}_attn_{depth}')(x)
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

    if 'imagenet' == weights and model_name in WEIGHT_URLS:
        weights_url = WEIGHT_URLS[model_name]
        weights_hash = WEIGHT_HASHES[model_name]
        weights_path = data_utils.get_file(origin=weights_url, file_hash=weights_hash, cache_subdir='coma')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if include_top:
        return model

    outputs = model.get_layer(name='norm').output
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def CoMATiny(embed_dim=64, stem_depth=1, stage_depths=(4, 4, 18, 2), path_drop=0.1, **kwargs):
    return CoMA(
        embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, path_drop=path_drop,
        model_name='coma-tiny', **kwargs)


def CoMASmall(embed_dim=96, stem_depth=2, stage_depths=(2, 6, 16, 3), path_drop=0.2, **kwargs):
    return CoMA(
        embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, path_drop=path_drop,
        model_name='coma-small', **kwargs)


def CoMABase(embed_dim=128, stem_depth=3, stage_depths=(2, 8, 20, 4), path_drop=0.3, **kwargs):
    return CoMA(
        embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, path_drop=path_drop,
        model_name='coma-base', **kwargs)


def CoMALarge(embed_dim=160, stem_depth=4, stage_depths=(2, 10, 24, 5), path_drop=0.4, **kwargs):
    return CoMA(
        embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, path_drop=path_drop,
        model_name='coma-large', **kwargs)

# Swin v2
#  96 / 2 2  6 2
#  96 / 2 2 18 2
# 128 / 2 2 18 2
# 192 / 2 2 18 2


# GCViT
#  64 / 3 4 19 5
#  96 / 3 4 19 5
# 128 / 3 4 19 5
# 192 / 3 4 19 5


# DaViT
#  96 / 2 2  6 2
#  96 / 2 2 18 2
# 128 / 2 2 18 2
# 192 / 2 2 18 2


# MaxViT
#  64 / 2 2  5 2
#  96 / 2 2  5 2
#  96 / 2 6 14 2
# 128 / 2 6 14 2


# CoAt
#  64 /  2 6 14 2
# 128 /  2 6 14 2
# 192 /  2 6 14 2
# 192 / 2 12 28 2


# EffNet
# 48 / 4 4 15 15
# 48 / 5 5 21 18
# 64 / 7 7 29 32


# T  28- 31 / 18G
# S  49- 69 / 36G
# B  87-120 / 47G-74G
# L 197-212 / 103G-133G

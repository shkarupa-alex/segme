import numpy as np
import tensorflow as tf
from keras import backend, initializers, layers, models
from keras.applications import imagenet_utils
from keras.applications.efficientnet_v2 import CONV_KERNEL_INITIALIZER
from keras.mixed_precision import global_policy
from keras.utils import data_utils, layer_utils
from segme.common.convnormact import Norm, ConvNormAct, ConvNorm, Conv, Act
from segme.common.drop import DropPath
from segme.common.mbconv import MBConv
from segme.common.grn import GRN
from segme.policy.backbone.diy.coma.attn import DHMSA, CHMSA, GGMSA

WEIGHT_URLS = {}
WEIGHT_HASHES = {}


# TODO: use_attn_dw, use_mlp_dw


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
            y = Conv(filters, 3, kernel_initializer=CONV_KERNEL_INITIALIZER, name=f'{name}_{i + 1}_conv')(x)
            y = Act(name=f'{name}_{i + 1}_act')(y)
            y = Norm(
                center=False, gamma_initializer=initializers.Constant(path_gamma[i]), name=f'{name}_{i + 1}_norm')(y)
            y = DropPath(path_drop[i], name=f'{name}_{i + 1}_drop')(y)
            x = layers.add([y, x], name=f'{name}_{i + 1}_add')

        return x

    return apply


def Reduce(fused, kernel_size=3, expand_ratio=3., name=None):
    if name is None:
        counter = backend.get_uid('reduce')
        name = f'reduce_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        expand_filters = int(channels * expand_ratio)

        # From EfficientNet2
        if fused:
            x = Conv(
                expand_filters, kernel_size, strides=2, kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=f'{name}_expand')(inputs)
        else:
            x = Conv(expand_filters, 1, use_bias=False, name=f'{name}_expand_pw')(inputs)
            x = Conv(
                None, kernel_size, strides=2, kernel_initializer=CONV_KERNEL_INITIALIZER, name=f'{name}_expand_dw')(x)
        x = Act(name=f'{name}_act')(x)
        x = GRN(center=False, name=f'{name}_grn')(x)  # From ConvNeXt2
        x = Conv(channels * 2, 1, use_bias=False, name=f'{name}_squeeze')(x)
        x = Norm(center=False, name=f'{name}_norm')(x)

        return x

    return apply


def MLPConv(fused, kernel_size=3, expand_ratio=3., path_drop=0., gamma_initializer='ones', name=None):
    if name is None:
        counter = backend.get_uid('mlpconv')
        name = f'mlpconv_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        expand_filters = int(channels * expand_ratio)

        # From EfficientNet2
        if fused:
            x = Conv(
                expand_filters, kernel_size, kernel_initializer=CONV_KERNEL_INITIALIZER, name=f'{name}_expand')(inputs)
        else:
            x = Conv(
                expand_filters, 1, use_bias=False, name=f'{name}_expand_pw')(inputs)
            x = Conv(None, kernel_size, kernel_initializer=CONV_KERNEL_INITIALIZER, name=f'{name}_expand_dw')(x)
        x = Act(name=f'{name}_act')(x)
        x = GRN(center=False, name=f'{name}_grn')(x)  # From ConvNeXt2
        x = Conv(channels, 1, use_bias=False, name=f'{name}_squeeze')(x)
        x = Norm(center=False, gamma_initializer=gamma_initializer, name=f'{name}_norm')(x)
        x = DropPath(path_drop, name=f'{name}_drop')(x)
        x = layers.add([x, inputs], name=f'{name}_add')

        return x

    return apply


def ConvBlock(fused, kernel_size=3, expand_ratio=3., path_gamma=1., path_drop=0., name=None):
    if name is None:
        counter = backend.get_uid('conv_block')
        name = f'conv_block_{counter}'

    gamma_initializer = initializers.Constant(path_gamma)

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        x = MLPConv(
            fused, kernel_size=kernel_size, expand_ratio=expand_ratio, path_drop=path_drop,
            gamma_initializer=gamma_initializer, name=f'{name}_mlpconv')(inputs)

        return x

    return apply


def WindBlock(current_window, pretrain_window, num_heads, dilation_rate=1, kernel_size=3, path_drop=0., expand_ratio=3.,
              path_gamma=1., name=None):
    if name is None:
        counter = backend.get_uid('attn_block')
        name = f'attn_block_{counter}'

    gamma_initializer = initializers.Constant(path_gamma)

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        x = DHMSA(
            current_window, pretrain_window, num_heads, dilation_rate=dilation_rate, name=f'{name}_dhmsa_attn')(inputs)
        x = Norm(center=False, gamma_initializer=gamma_initializer, name=f'{name}_dhmsa_norm')(x)
        x = DropPath(path_drop, name=f'{name}_dhmsa_drop')(x)
        x = layers.add([x, inputs], name=f'{name}_dhmsa_add')

        x = MLPConv(
            False, kernel_size=kernel_size, expand_ratio=expand_ratio, path_drop=path_drop,
            gamma_initializer=gamma_initializer, name=f'{name}_mlpconv')(x)

        return x

    return apply


def ChanBlock(num_heads, kernel_size=3, path_drop=0., expand_ratio=3., path_gamma=1., name=None):
    if name is None:
        counter = backend.get_uid('attn_block')
        name = f'attn_block_{counter}'

    gamma_initializer = initializers.Constant(path_gamma)

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        x = CHMSA(num_heads, name=f'{name}_chmsa_attn')(inputs)
        x = Norm(center=False, gamma_initializer=gamma_initializer, name=f'{name}_chmsa_norm')(x)
        x = DropPath(path_drop, name=f'{name}_chmsa_drop')(x)
        x = layers.add([x, inputs], name=f'{name}_chmsa_add')

        x = MLPConv(
            False, kernel_size=kernel_size, expand_ratio=expand_ratio, path_drop=path_drop,
            gamma_initializer=gamma_initializer, name=f'{name}_mlpconv')(x)

        return x

    return apply


def GridBlock(current_window, pretrain_window, num_heads, kernel_size=3, path_drop=0., expand_ratio=3., path_gamma=1.,
              name=None):
    if name is None:
        counter = backend.get_uid('attn_block')
        name = f'attn_block_{counter}'

    gamma_initializer = initializers.Constant(path_gamma)

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        x = GGMSA(current_window, pretrain_window, num_heads, name=f'{name}_ggmsa_attn')(inputs)
        x = Norm(center=False, gamma_initializer=gamma_initializer, name=f'{name}_ggmsa_norm')(x)
        x = DropPath(path_drop, name=f'{name}_ggmsa_drop')(x)
        x = layers.add([x, inputs], name=f'{name}_ggmsa_add')

        x = MLPConv(
            False, kernel_size=kernel_size, expand_ratio=expand_ratio, path_drop=path_drop,
            gamma_initializer=gamma_initializer, name=f'{name}_mlp')(x)

        return x

    return apply


def CoMA(
        embed_dim, stem_depth, stage_depths, current_window=8, pretrain_window=8, expand_ratio=3, path_gamma=0.01,
        path_drop=0.2, pretrain_size=384, input_shape=None, include_top=True, model_name='coma', pooling=None,
        weights=None, input_tensor=None, classes=1000, classifier_activation='softmax', include_preprocessing=False):
    """ Inspired with:

    02.01.2023 ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
        + global response normalization
    15.11.2022 Focal Modulation Networks
        + overlapped patch embedding
        + deeper but thinner
        - focal-modulation instead of self-attention
        - context aggregation
    10.11.2022 Demystify Transformers & Convolutions in Modern Image Deep Networks
        + overlapped patch embedding and reduction
        + haloing for local-attention spatial token mixer
    24.10.2022 MetaFormer Baselines for Vision
        + stage architecture CCTT
        ~ disable all biases
        ~ scaling the residual branch
        ~ stage ratio 1:4:6:1
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
        ~ stage ratio 1:1:9:1
        - grid self-attention
    16.05.2022 Activating More Pixels in Image Super-Resolution Transformer
        + overlapping window cross-attention
        ~ channel attention
        ! enlarging window size of self-attention
    11.04.2022 Swin Transformer V2: Scaling Up Capacity and Resolution
        + log-spaced continuous position bias
        + residual post normalization
        + scaled cosine attention
    07.04.2022 DaViT: Dual Attention Vision Transformers
        + channel group self-attention
    02.03.2022 A ConvNet for the 2020s
        + adding normalization layers wherever spatial resolution is changed
        ~ stage ratio 1:1:9:1
    28.10.2021 SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
        + overlapped reduction
        + depthwise convolution in MLP
        - reduce length for efficient self-attention
    24.10.2021 Leveraging Batch Normalization for Vision Transformers
        - BN is faster than LN in early stages when input has larger spatial resolution and smaller channel number
        - BN in MLP
        - BN in attention
    15.09.2021 CoAtNet: Marrying Convolution and Attention for All Data Sizes
        + stage architecture CCTT
        ~ MBConv for reduction and as convolutional block
        + stride-2 stem
        ~ stage ratio 1:3:7:1
    07.06.2021 Scaling Local Self-Attention for Parameter Efficient Visual Backbones
        + overlapping window with halo = 1/2 of window size
        + stage architecture CCTT
        ! accuracy consistently improves as the window size increases
    29.03.2021 CvT: Introducing Convolutions to Vision Transformers
        + depthwise convolution in attention projection
        ~ overlapped patch embedding and reduction
        - query & key projection with stride 2
    13.03.2021 Revisiting ResNets: Improved Training and Scaling Strategies
        ~ zero-gamma trick
        - average pooling & pointwise convolution as main reduction branch
        ! decreasing weight decay when using augmentations

    27.05.2022 Revealing the Dark Secrets of Masked Image Modeling
        * MIM pretraining is better for downstream tasks
    """
    if embed_dim % 32:
        raise ValueError('Embedding size should be a multiple of 32.')

    if len(stage_depths) < 4:
        raise ValueError('Number of stages should be greater then 4.')

    if weights not in {'imagenet', None} and not tf.io.gfile.exists(weights):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes not in {1000, 14607}:
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

    path_gammas = np.linspace(path_gamma, 1e-5, stem_depth + sum(stage_depths) + len(stage_depths)).tolist()
    path_drops = np.linspace(0., path_drop, stem_depth + sum(stage_depths) + len(stage_depths)).tolist()

    stem_gammas, path_gammas = path_gammas[:stem_depth], path_gammas[stem_depth:]
    stem_drops, path_drops = path_drops[:stem_depth], path_drops[stem_depth:]
    x = Stem(embed_dim // 2, stem_depth, path_gamma=stem_gammas, path_drop=stem_drops, name='stem')(x)
    x = layers.Activation('linear', name='stem_out')(x)

    for i, stage_depth in enumerate(stage_depths):
        fused = 0 == i
        num_heads = embed_dim // 2 ** (5 - i)

        stage_gammas, path_gammas = path_gammas[:stage_depth + 1], path_gammas[stage_depth + 1:]
        stage_drops, path_drops = path_drops[:stage_depth + 1], path_drops[stage_depth + 1:]

        # From GCViT
        x = Reduce(fused, expand_ratio=expand_ratio, name=f'stage_{i}_reduce')(x)

        for j in range(stage_depth):
            if i < 2:  # From CoAtNet
                kernel_size = 3 if fused else 5
                x = ConvBlock(
                    fused, kernel_size=kernel_size, expand_ratio=expand_ratio, path_gamma=stage_gammas[j],
                    path_drop=stage_drops[j], name=f'stage_{i}_conv_{j}')(x)
                continue

            current_size = pretrain_size // 2 ** (i + 2)
            dilation_max = current_size * 2 // (3 * current_window)
            dilation_rate = 1 + (j % 2) * (j // 2 % max(1, dilation_max - 1) + 1)
            dilation_rate = min(dilation_rate, dilation_max)
            dilation_rate = max(dilation_rate, 1)

            x = WindBlock(
                current_window, pretrain_window, num_heads, dilation_rate=dilation_rate, expand_ratio=expand_ratio,
                path_gamma=stage_gammas[j], path_drop=stage_drops[j], name=f'stage_{i}_attn_{j}')(x)

        # From DaViT, HAT
        x = ChanBlock(
            num_heads, path_gamma=stage_gammas[stage_depth], expand_ratio=expand_ratio,
            path_drop=stage_drops[stage_depth], name=f'stage_{i}_attn_{stage_depth}')(x)
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


def CoMATiny(embed_dim=64, stem_depth=2, stage_depths=(3, 4, 19, 3), path_drop=0.1, **kwargs):
    # 26.6 M, 18.5 G
    return CoMA(
        embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, path_drop=path_drop,
        model_name='coma-tiny', **kwargs)


def CoMASmall(embed_dim=96, stem_depth=2, stage_depths=(3, 4, 19, 3), **kwargs):
    # 58.9 M, 39.4 G
    return CoMA(
        embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, model_name='coma-small', **kwargs)


def CoMABase(embed_dim=128, stem_depth=3, stage_depths=(4, 5, 21, 3), **kwargs):
    # 110.2 M, 78.6 G
    return CoMA(
        embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, model_name='coma-base', **kwargs)


def CoMALarge(embed_dim=160, stem_depth=4, stage_depths=(5, 6, 21, 5), **kwargs):
    # 205.9 M, 136.3 G
    return CoMA(
        embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, model_name='coma-large', **kwargs)

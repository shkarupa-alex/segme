import numpy as np
import tensorflow as tf
from keras import backend, initializers, layers, models
from keras.mixed_precision import global_policy
from keras.src.applications import imagenet_utils
from keras.src.applications.efficientnet_v2 import CONV_KERNEL_INITIALIZER
from keras.src.utils import data_utils, layer_utils
from segme.common.convnormact import Norm, Conv, Act
from segme.common.drop import SlicePath, RestorePath
from segme.common.grn import GRN
from segme.common.attn import SwinAttention, SlideAttention
from segme.model.classification.data import tree_class_map
from segme.policy import cnapol

WEIGHT_URLS = {}
WEIGHT_HASHES = {}


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
            y, mask = SlicePath(path_drop[i], name=f'{name}_{i + 1}_slice')(x)
            y = Conv(filters, 3, kernel_initializer=CONV_KERNEL_INITIALIZER, name=f'{name}_{i + 1}_conv')(y)
            y = Act(name=f'{name}_{i + 1}_act')(y)
            y = Norm(
                center=False, gamma_initializer=initializers.Constant(path_gamma[i]), name=f'{name}_{i + 1}_norm')(y)
            y = RestorePath(path_drop[i], name=f'{name}_{i + 1}_drop')([y, mask])
            x = layers.add([y, x], name=f'{name}_{i + 1}_add')

        return x

    return apply


def Reduce(filters, fused=False, kernel_size=3, expand_ratio=3., name=None):
    if name is None:
        counter = backend.get_uid('reduce')
        name = f'reduce_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        expand_filters = int(channels * expand_ratio)
        if expand_filters < filters:
            raise ValueError('Expansion size must be greater or equal to output one.')

        if fused:  # From EfficientNet2
            x = Conv(
                expand_filters, kernel_size, strides=2, kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=f'{name}_expand')(inputs)
        else:
            x = Conv(expand_filters, 1, use_bias=False, name=f'{name}_expand_pw')(inputs)
            x = Conv(
                None, kernel_size, strides=2, kernel_initializer=CONV_KERNEL_INITIALIZER, name=f'{name}_expand_dw')(x)

        x = Act(name=f'{name}_act')(x)

        if expand_ratio > 2:
            x = GRN(center=False, name=f'{name}_grn')(x)  # From ConvNeXt2

        x = Conv(filters, 1, use_bias=False, name=f'{name}_squeeze')(x)
        x = Norm(center=False, name=f'{name}_norm')(x)

        return x

    return apply


def MLPConv(filters, kernel_size=3, expand_ratio=3., path_drop=0., gamma_initializer='ones', name=None):
    if name is None:
        counter = backend.get_uid('mlpconv')
        name = f'mlpconv_{counter}'

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        expand_filters = int(channels * expand_ratio)

        if filters == channels:
            x, mask = SlicePath(path_drop, name=f'{name}_slice')(inputs)
        else:
            x = inputs

        x = Conv(expand_filters, 1, use_bias=False, name=f'{name}_expand_pw')(x)
        x = Conv(None, kernel_size, kernel_initializer=CONV_KERNEL_INITIALIZER, name=f'{name}_expand_dw')(x)
        x = Act(name=f'{name}_act')(x)

        if filters == channels and expand_ratio > 2:
            x = GRN(center=False, name=f'{name}_grn')(x)  # From ConvNeXt2

        x = Conv(filters, 1, use_bias=False, name=f'{name}_squeeze')(x)

        if filters == channels:
            x = Norm(center=False, gamma_initializer=gamma_initializer, name=f'{name}_norm')(x)
            x = RestorePath(path_drop, name=f'{name}_drop')([x, mask])
            x = layers.add([x, inputs], name=f'{name}_add')
        else:
            x = Norm(center=False, name=f'{name}_norm')(x)

        return x

    return apply


def LocalBlock(
        filters, window_size, num_heads, qk_units=16, dilation_rate=1, kernel_size=3, expand_ratio=3., path_gamma=1.,
        path_drop=0., name=None):
    if name is None:
        counter = backend.get_uid('attn_block')
        name = f'attn_block_{counter}'

    gamma_initializer = initializers.Constant(path_gamma)

    def apply(inputs):
        with cnapol.policy_scope('conv-ln-gelu'):
            channels = inputs.shape[-1]
            if channels is None:
                raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

            x, mask = SlicePath(path_drop, name=f'{name}_slide_slice')(inputs)
            x = SlideAttention(
                window_size, num_heads, qk_units=qk_units, cpb_units=num_heads * 8, proj_bias=False,
                dilation_rate=dilation_rate, name=f'{name}_slide_attn')(x)
            x = Norm(center=False, gamma_initializer=gamma_initializer, name=f'{name}_slide_norm')(x)
            x = RestorePath(path_drop, name=f'{name}_slide_drop')([x, mask])
            x = layers.add([x, inputs], name=f'{name}_slide_add')

            x = MLPConv(
                filters, kernel_size=kernel_size, expand_ratio=expand_ratio, path_drop=path_drop,
                gamma_initializer=gamma_initializer, name=f'{name}_mlpconv')(x)

            return x

    return apply


def SwinBlock(current_window, pretrain_window, num_heads, shift_mode, qk_units=16, kernel_size=3, path_drop=0.,
              expand_ratio=3., path_gamma=1., name=None):
    if name is None:
        counter = backend.get_uid('attn_block')
        name = f'attn_block_{counter}'

    gamma_initializer = initializers.Constant(path_gamma)

    def apply(inputs):
        with cnapol.policy_scope('conv-ln-gelu'):
            channels = inputs.shape[-1]
            if channels is None:
                raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

            x, mask = SlicePath(path_drop, name=f'{name}_swin_slice')(inputs)
            x = SwinAttention(
                current_window, pretrain_window, num_heads, qk_units=qk_units, cpb_units=num_heads * 8, proj_bias=False,
                shift_mode=shift_mode, name=f'{name}_swin_attn')(x)
            x = Norm(center=False, gamma_initializer=gamma_initializer, name=f'{name}_swin_norm')(x)
            x = RestorePath(path_drop, name=f'{name}_swin_drop')([x, mask])
            x = layers.add([x, inputs], name=f'{name}_swin_add')

            x = MLPConv(
                False, kernel_size=kernel_size, expand_ratio=expand_ratio, path_drop=path_drop,
                gamma_initializer=gamma_initializer, name=f'{name}_mlpconv')(x)

            return x

    return apply


def CoMA(
        stem_dim, stem_depth, embed_dim, stage_depths, current_window=8, pretrain_window=8, qk_units=16,
        path_gamma=0.01, path_drop=0.2, pretrain_size=384, input_shape=None, include_top=True, model_name='coma',
        pooling=None, weights=None, input_tensor=None, classes=1000, classifier_activation='softmax',
        include_preprocessing=False):
    """ Inspired with:

    09.06.2023 FasterViT: Fast Vision Transformers with Hierarchical Attention
        - ?
    11.05.2023 EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention
        + overlapping patch embedding
        ~ Q, K (=16) and MLP (=x2) dimensions are largely trimmed for late stages
        ~ depthwise convolution over q and before MLP
        - fewer attention blocks (more MLPs)
        - feeding each head with only a split of the full features
    14.04.2023 DINOv2: Learning Robust Visual Features without Supervision
        + efficient stochastic depth (slice instead of drop)
        - fast and memory-efficient FlashAttention
    09.04.2023 Slide-Transformer: Hierarchical Vision Transformer with Local Self-Attention
        + depthwise convolution with fixed weights instead of im2col
        + depthwise convolution with learnable weights for deformable attention
        + local attention with kernel size 3 for stage 1 & 2
    02.01.2023 ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
        + global response normalization
    23.12.2022 A Close Look at Spatial Modeling: From Attention to Convolution
        + depthwise convolution in MLP
        ~ overlapped patch embedding (7-4 for stem, 3-2 for downsampling)
        - global context with avg pool self attention
    15.11.2022 Focal Modulation Networks
        + overlapped patch embedding
        + deeper but thinner
        - focal-modulation instead of self-attention
        - context aggregation
    10.11.2022 Demystify Transformers & Convolutions in Modern Image Deep Networks
        + overlapped patch embedding and reduction
        - haloing for local-attention spatial token mixer
    24.10.2022 MetaFormer Baselines for Vision
        ~ disable all biases
        ~ scaling the residual branch
        ~ stage ratio 1:4:6:1
        - stage architecture CCTT
        - convolutional block with separated spatial mixer & MLP
        - StarReLU with learnable scale and bias
    01.10.2022 Global Context Vision Transformers
        + stride-2 stem
        ~ modified Fused-MBConv block for reduction
        ~ stage ratio 3:4:19:5
        - global query generation
    29.09.2022 Dilated Neighborhood Attention Transformer
        ? dilated (sparse) window self-attention
        ? gradual dilation order (1, 2, 1, 4, 1, 6)
    09.09.2022 MaxViT: Multi-Axis Vision Transformer
        ~ stage ratio 1:1:9:1
        - grid self-attention
    16.05.2022 Activating More Pixels in Image Super-Resolution Transformer
        - overlapping window cross-attention
        - channel attention
        - enlarging window size of self-attention
    11.04.2022 Swin Transformer V2: Scaling Up Capacity and Resolution
        + log-spaced continuous position bias
        + residual post normalization
        + scaled cosine attention
    07.04.2022 DaViT: Dual Attention Vision Transformers
        - channel group self-attention
    02.03.2022 A ConvNet for the 2020s
        + adding normalization layers wherever spatial resolution is changed
        ~ stage ratio 1:1:9:1
    28.10.2021 SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
        + overlapped reduction
        + depthwise convolution in MLP
        - reduce length for efficient self-attention
    24.10.2021 Leveraging Batch Normalization for Vision Transformers
        - BN in MLP
        - BN in attention
        ! BN is faster than LN in early stages when input has larger spatial resolution and smaller channel number
    15.09.2021 CoAtNet: Marrying Convolution and Attention for All Data Sizes
        + stride-2 stem
        ~ MBConv for reduction and as convolutional block
        ~ stage ratio 1:3:7:1
        - stage architecture CCTT
    23.06.2021 EfficientNetV2: Smaller Models and Faster Training
        + depthwise convolutions are slow in early layers but effective in later
        + non-uniform capacity scaling
        ! adjusts regularization according to image size
    07.06.2021 Scaling Local Self-Attention for Parameter Efficient Visual Backbones
        - overlapping window with halo = 1/2 of window size
        - stage architecture CCTT
        ! accuracy consistently improves as the window size increases
    29.03.2021 CvT: Introducing Convolutions to Vision Transformers
        + overlapped patch embedding and reduction
        - depthwise convolution in attention projection
        - query & key projection with stride 2
    13.03.2021 Revisiting ResNets: Improved Training and Scaling Strategies
        ~ zero-gamma trick
        - average pooling & pointwise convolution as main reduction branch
        ! decreasing weight decay when using augmentations
    27.05.2022 Revealing the Dark Secrets of Masked Image Modeling
        ! MIM pretraining is better for downstream tasks
    """
    if embed_dim % 32:
        raise ValueError('Embedding size should be a multiple of 32.')

    if len(stage_depths) < 4:
        raise ValueError('Number of stages should be greater then 4.')

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
    pretrain_size = pretrain_size or pretrain_window * min_size # TODO
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

    path_gammas = np.linspace(path_gamma, 1e-5, stem_depth + sum(stage_depths)).tolist()
    path_drops = np.linspace(0., path_drop, stem_depth + sum(stage_depths)).tolist()

    with cnapol.policy_scope('conv-gn-gelu'):
        stem_gammas, path_gammas = path_gammas[:stem_depth], path_gammas[stem_depth:]
        stem_drops, path_drops = path_drops[:stem_depth], path_drops[stem_depth:]
        x = Stem(stem_dim, stem_depth, path_gamma=stem_gammas, path_drop=stem_drops, name='stem')(x)
        x = layers.Activation('linear', name='stem_out')(x)

    with cnapol.policy_scope('conv-ln-gelu'):
        for i, stage_depth in enumerate(stage_depths):
            expand_ratio = 2 + int(i >= 2)
            stage_gammas, path_gammas = path_gammas[:stage_depth], path_gammas[stage_depth:]
            stage_drops, path_drops = path_drops[:stage_depth], path_drops[stage_depth:]

            if stage_depth >= 8:
                curr_dim = embed_dim * 0.75 * 2 ** i
            else:
                curr_dim = embed_dim * 2 ** i

            # From EfficientNet2
            x = Reduce(curr_dim, fused=0 == i, expand_ratio=expand_ratio, name=f'stage_{i}_reduce')(x)

            for j in range(stage_depth):
                num_heads = embed_dim // 2 ** (5 - i)

                # if j < 2:
                #     x = LocalBlock()

                current_size = pretrain_size // 2 ** (i + 2)
                dilation_max = current_size * 2 // (3 * current_window)
                dilation_rate = 1 + (j % 2) * (j // 2 % max(1, dilation_max - 1) + 1)
                dilation_rate = min(dilation_rate, dilation_max)
                dilation_rate = max(dilation_rate, 1)

                x = SwinBlock(
                    current_window, pretrain_window, num_heads, dilation_rate=dilation_rate, expand_ratio=expand_ratio,
                    path_gamma=stage_gammas[j], path_drop=stage_drops[j], name=f'stage_{i}_attn_{j}')(x)
                # TODO: remove the last exitation in last block

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

    weights_key = f'{model_name}_{cnapol.global_policy()}'
    if 'imagenet' == weights and weights_key in WEIGHT_URLS:
        weights_url = WEIGHT_URLS[weights_key]
        weights_hash = WEIGHT_HASHES[weights_key]
        weights_path = data_utils.get_file(origin=weights_url, file_hash=weights_hash, cache_subdir='coma')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if include_top:
        return model

    outputs = model.get_layer(name='norm').output
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


# def CoMATiny(stem_dim=16, stem_depth=2, embed_dim=64, stage_depths=(3, 3, 21, 3), path_drop=0.1, **kwargs):
#     #
#     return CoMA(
#         embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, path_drop=path_drop,
#         model_name='coma_tiny', **kwargs)


def CoMASmall(stem_dim=24, stem_depth=2, embed_dim=48, stage_depths=(3, 3, 18, 3), **kwargs):
    #
    config = [
        'r2_s1_e1_o24_c1',
        'r4_s2_e4_o48_c1',
        'r4_s2_e4_o64_c1',
        'r6_s2_e4_o128_se0.25',
        'r9_s1_e6_o160_se0.25',
        'r15_s2_e6_o256_se0.25',
    ]
    return CoMA(
        embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, model_name='coma_small', **kwargs)

# def CoMABase(stem_dim=40, stem_depth=3, embed_dim=128, stage_depths=(3, 3, 21, 3), **kwargs):
#     #
#     return CoMA(
#         embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, model_name='coma_base', **kwargs)
#
#
# def CoMALarge(stem_dim=64, stem_depth=4, embed_dim=160, stage_depths=(3, 3, 21, 3), **kwargs):
#     #
#     return CoMA(
#         embed_dim=embed_dim, stem_depth=stem_depth, stage_depths=stage_depths, model_name='coma_large', **kwargs)

# huge stem=4

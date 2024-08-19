import numpy as np
from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import models
from keras.src.applications import imagenet_utils
from keras.src.applications.efficientnet_v2 import CONV_KERNEL_INITIALIZER
from keras.src.dtype_policies import dtype_policy
from keras.src.ops import operation_utils
from keras.src.utils import file_utils
from keras.src.utils import naming

from segme.common.attn.slide import SlideAttention
from segme.common.attn.swin import SwinAttention
from segme.common.convnormact import Act
from segme.common.convnormact import Conv
from segme.common.convnormact import Norm
from segme.common.drop import DropPath
from segme.common.grn import GRN
from segme.common.mapool import MultiHeadAttentionPooling
from segme.common.simpool import SimPool
from segme.policy import cnapol

WEIGHT_URLS = {}
WEIGHT_HASHES = {}


def Stem(filters, depth, path_drop=0.0, path_gamma=1.0, name=None):
    if name is None:
        counter = naming.get_uid("stem")
        name = f"stem_{counter}"

    if isinstance(path_gamma, float):
        path_gamma = [path_gamma] * depth
    if len(path_gamma) != depth:
        raise ValueError("Number of path gammas must equals to depth.")

    if isinstance(path_drop, float):
        path_drop = [path_drop] * depth
    if len(path_drop) != depth:
        raise ValueError("Number of path dropouts must equals to depth.")

    def apply(inputs):
        x = Conv(
            filters,
            3,
            strides=2,
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=f"{name}_0_conv",
        )(inputs)
        x = Act(name=f"{name}_0_act")(x)
        x = Norm(center=False, name=f"{name}_0_norm")(x)

        for i in range(depth):
            y = Conv(
                filters,
                3,
                use_bias=False,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=f"{name}_{i + 1}_conv",
            )(x)
            y = Act(name=f"{name}_{i + 1}_act")(y)
            y = Norm(
                center=False,
                gamma_initializer=initializers.Constant(path_gamma[i]),
                name=f"{name}_{i + 1}_norm",
            )(y)
            y = DropPath(path_drop[i], name=f"{name}_{i + 1}_drop")(y)
            x = layers.add([y, x], name=f"{name}_{i + 1}_add")

        return x

    return apply


def Reduce(filters, fused=False, kernel_size=3, expand_ratio=4.0, name=None):
    if name is None:
        counter = naming.get_uid("reduce")
        name = f"reduce_{counter}"

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        expand_filters = int(channels * expand_ratio)
        if expand_filters < filters:
            raise ValueError(
                "Expansion size must be greater or equal to output one."
            )

        if fused:  # From EfficientNet2
            x = Conv(
                expand_filters,
                kernel_size,
                strides=2,
                use_bias=False,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=f"{name}_expand",
            )(inputs)
        else:
            x = Conv(
                expand_filters, 1, use_bias=False, name=f"{name}_expand_pw"
            )(inputs)
            x = Conv(
                None,
                kernel_size,
                strides=2,
                use_bias=False,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=f"{name}_expand_dw",
            )(x)

        x = Act(name=f"{name}_act")(x)
        x = Conv(filters, 1, use_bias=False, name=f"{name}_squeeze")(x)
        x = Norm(center=False, name=f"{name}_norm")(x)

        return x

    return apply


def MLPConv(
    filters,
    fused,
    kernel_size=3,
    expand_ratio=3.0,
    path_drop=0.0,
    gamma_initializer="ones",
    name=None,
):
    if name is None:
        counter = naming.get_uid("mlpconv")
        name = f"mlpconv_{counter}"

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        if expand_ratio < 1.0:
            raise ValueError("Expansion ratio must be greater or equal to 1.")
        expand_filters = int(channels * expand_ratio)

        if fused:
            x = Conv(
                expand_filters,
                kernel_size,
                use_bias=False,
                name=f"{name}_expand",
            )(inputs)
        else:
            x = Conv(
                expand_filters, 1, use_bias=False, name=f"{name}_expand_pw"
            )(inputs)
            x = Conv(
                None,
                kernel_size,
                use_bias=False,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=f"{name}_expand_dw",
            )(x)
        x = Act(name=f"{name}_act")(x)

        if filters == channels and expand_ratio > 2.0:
            x = GRN(center=False, name=f"{name}_grn")(x)  # From ConvNeXt2

        x = Conv(filters, 1, use_bias=False, name=f"{name}_squeeze")(x)

        if filters == channels:
            x = Norm(
                center=False,
                gamma_initializer=gamma_initializer,
                name=f"{name}_norm",
            )(x)
            x = DropPath(path_drop, name=f"{name}_drop")(x)
            x = layers.add([x, inputs], name=f"{name}_add")
        else:
            x = Norm(center=False, name=f"{name}_norm")(x)

        return x

    return apply


def SwinBlock(
    filters,
    current_window,
    pretrain_window,
    num_heads,
    shift_mode,
    qk_units=16,
    kernel_size=3,
    path_drop=0.0,
    expand_ratio=3.0,
    path_gamma=1.0,
    name=None,
):
    if name is None:
        counter = naming.get_uid("attn_block")
        name = f"attn_block_{counter}"

    gamma_initializer = initializers.Constant(path_gamma)

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        x = SwinAttention(
            current_window,
            pretrain_window,
            num_heads,
            qk_units=qk_units,
            cpb_units=num_heads * 8,
            proj_bias=False,
            shift_mode=shift_mode,
            name=f"{name}_swin_attn",
        )(inputs)
        x = Norm(
            center=False,
            gamma_initializer=gamma_initializer,
            name=f"{name}_swin_norm",
        )(x)
        x = DropPath(path_drop, name=f"{name}_swin_drop")(x)
        x = layers.add([x, inputs], name=f"{name}_swin_add")

        x = MLPConv(
            filters,
            False,
            kernel_size=kernel_size,
            expand_ratio=expand_ratio,
            path_drop=path_drop,
            gamma_initializer=gamma_initializer,
            name=f"{name}_mlpconv",
        )(x)

        return x

    return apply


def LocalBlock(
    filters,
    window_size,
    num_heads,
    qk_units=16,
    dilation_rate=1,
    kernel_size=3,
    path_drop=0.0,
    expand_ratio=3.0,
    path_gamma=1.0,
    name=None,
):
    if name is None:
        counter = naming.get_uid("attn_block")
        name = f"attn_block_{counter}"

    gamma_initializer = initializers.Constant(path_gamma)

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        x = SlideAttention(
            window_size,
            num_heads,
            qk_units=qk_units,
            cpb_units=num_heads * 8,
            proj_bias=False,
            dilation_rate=dilation_rate,
            name=f"{name}_slide_attn",
        )(inputs)
        x = Norm(
            center=False,
            gamma_initializer=gamma_initializer,
            name=f"{name}_slide_norm",
        )(x)
        x = DropPath(path_drop, name=f"{name}_slide_drop")(x)
        x = layers.add([x, inputs], name=f"{name}_slide_add")

        x = MLPConv(
            filters,
            False,
            kernel_size=kernel_size,
            expand_ratio=expand_ratio,
            path_drop=path_drop,
            gamma_initializer=gamma_initializer,
            name=f"{name}_mlpconv",
        )(x)

        return x

    return apply


def CoMA(
    stem_dim,
    stem_depth,
    embed_dim,
    stage_depths,
    pretrain_window,
    current_window=None,
    path_gamma=0.01,
    path_drop=0.0,
    pretrain_size=384,
    current_size=None,
    input_shape=None,
    include_top=True,
    model_name="coma",
    pooling=None,
    weights=None,
    input_tensor=None,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    """Inspired with:

    09.06.2023 FasterViT: Fast Vision Transformers with Hierarchical Attention
        ! initial layers are memory-bound and better for compute-intensive
          operations, such as dense convolution
        ! later layers are math-limited and better for layer normalization,
          squeeze-and-excitation or attention
        ~ stride-2 stem with bn and relu
        - stage architecture CCTT
        - windows interaction through attention with averaged windows
    11.05.2023 EfficientViT: Memory Efficient Vision Transformer with Cascaded
      Group Attention
        + overlapping patch embedding
        ~ Q, K (=16) and MLP (=x2) dimensions are largely trimmed for late
          stages
        ~ depth-wise convolution over q and before MLP
        - fewer attention blocks (more MLPs)
        - feeding each head with only a split of the full features
    14.04.2023 DINOv2: Learning Robust Visual Features without Supervision
        ? efficient stochastic depth (slice instead of drop) // not working
          with XLA
        - fast and memory-efficient FlashAttention
    09.04.2023 Slide-Transformer: Hierarchical Vision Transformer with Local
      Self-Attention
        + depth-wise convolution with fixed weights instead of im2col
        + depth-wise convolution with learnable weights for deformable attention
        ~ local attention with kernel size 3 for stage 1 & 2
    02.01.2023 ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
      Autoencoders
        + global response normalization
    23.12.2022 A Close Look at Spatial Modeling: From Attention to Convolution
        + depth-wise convolution in MLP
        ~ overlapped patch embedding (7-4 for stem, 3-2 for downsampling)
        - global context with avg pool self attention
    15.11.2022 Focal Modulation Networks
        + overlapped patch embedding
        + deeper but thinner
        - focal-modulation instead of self-attention
        - context aggregation
    10.11.2022 Demystify Transformers & Convolutions in Modern Image Deep
      Networks
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
        - dilated (sparse) window self-attention
        - gradual dilation order (1, 2, 1, 4, 1, 6)
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
    28.10.2021 SegFormer: Simple and Efficient Design for Semantic Segmentation
      with Transformers
        + overlapped reduction
        + depth-wise convolution in MLP
        - reduce length for efficient self-attention
    24.10.2021 Leveraging Batch Normalization for Vision Transformers
        ! BN is faster than LN in early stages when input has larger spatial
          resolution and smaller channel number
        - BN in MLP
        - BN in attention
    15.09.2021 CoAtNet: Marrying Convolution and Attention for All Data Sizes
        + stride-2 stem
        ~ MBConv for reduction and as convolutional block
        ~ stage ratio 1:3:7:1
        - stage architecture CCTT
    23.06.2021 EfficientNetV2: Smaller Models and Faster Training
        ! depth-wise convolutions are slow in early layers but effective in
          later
        ! adjusts regularization according to image size
        ~ non-uniform capacity scaling
    07.06.2021 Scaling Local Self-Attention for Parameter Efficient Visual
      Backbones
        ! accuracy consistently improves as the window size increases
        - overlapping window with halo = 1/2 of window size
        - stage architecture CCTT
    29.03.2021 CvT: Introducing Convolutions to Vision Transformers
        + overlapped patch embedding and reduction
        - depth-wise convolution in attention projection
        - query & key projection with stride 2
    13.03.2021 Revisiting ResNets: Improved Training and Scaling Strategies
        ! decreasing weight decay when using augmentations
        + zero-gamma trick
        - average pooling & pointwise convolution as main reduction branch
    27.05.2022 Revealing the Dark Secrets of Masked Image Modeling
        ! MIM pretraining is better for downstream tasks
    """

    if embed_dim % 32:
        raise ValueError("Embedding size should be a multiple of 32.")

    if len(stage_depths) < 4:
        raise ValueError("Number of stages should be greater then 4.")

    current_window = current_window or pretrain_window

    if input_tensor is not None:
        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(
                f"Expecting `input_tensor` to be a symbolic tensor instance. "
                f"Got {input_tensor} of type {type(input_tensor)}"
            )

    # Determine proper input shape
    min_size = 2 ** (len(stage_depths) + 1)
    pretrain_size = pretrain_size or pretrain_window * min_size
    current_size = current_size or pretrain_size
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=current_size,
        min_size=min_size,
        data_format="channel_last",
        require_flatten=False,
        weights=weights,
    )
    input_dtype = dtype_policy.dtype_policy().compute_dtype

    if input_tensor is not None:
        if backend.is_keras_tensor(input_tensor):
            image = input_tensor
        else:
            image = layers.Input(
                shape=input_shape,
                name="images",
                dtype=input_dtype,
                tensor=input_tensor,
            )
    else:
        image = layers.Input(
            shape=input_shape, name="images", dtype=input_dtype
        )

    x = image

    if include_preprocessing:
        x = layers.Normalization(
            mean=np.array([0.485, 0.456, 0.406], "float32") * 255.0,
            variance=(np.array([0.229, 0.224, 0.225], "float32") * 255.0) ** 2,
            name="normalize",
        )(x)

    path_gammas = np.linspace(
        path_gamma, 1e-5, stem_depth + sum(stage_depths)
    ).tolist()
    path_drops = np.linspace(
        0.0, path_drop, stem_depth + sum(stage_depths)
    ).tolist()

    stem_drops, path_drops = path_drops[:stem_depth], path_drops[stem_depth:]
    stem_gammas, path_gammas = (
        path_gammas[:stem_depth],
        path_gammas[stem_depth:],
    )
    x = Stem(
        stem_dim,
        stem_depth,
        path_drop=stem_drops,
        path_gamma=stem_gammas,
        name="stem",
    )(x)
    x = layers.Activation("linear", name="stem_out")(x)

    for i, stage_depth in enumerate(stage_depths):
        stage_dims = embed_dim * 2**i
        if stage_depth >= sum(stage_depths) / 2:
            stage_dims = int(stage_dims * 0.75 / 32) * 32, stage_dims
        else:
            stage_dims = stage_dims, stage_dims
        num_heads = stage_dims[0] // 32

        stage_window = min(current_window, current_size // 2 ** (i + 2))
        expand_ratio = max(2, i + 1)
        stage_drops, path_drops = (
            path_drops[:stage_depth],
            path_drops[stage_depth:],
        )
        stage_gammas, path_gammas = (
            path_gammas[:stage_depth],
            path_gammas[stage_depth:],
        )

        # From EfficientNet2
        x = Reduce(stage_dims[0], fused=0 == i, name=f"stage_{i}_reduce")(x)

        for j in range(stage_depth):
            if stage_depth >= sum(stage_depths) / 2 and j > stage_depth / 2:
                stage_dim = stage_dims[1]
            else:
                stage_dim = stage_dims[0]

            if i + 1 == len(stage_depths) and j + 1 == stage_depth:
                # Remove squeezing in last block of the last stage
                stage_dim = int(stage_dims[1] * expand_ratio)

            if 0 == j % 3:
                x = SwinBlock(
                    stage_dim,
                    stage_window,
                    pretrain_window,
                    num_heads,
                    0,
                    path_drop=stage_drops[j],
                    expand_ratio=expand_ratio,
                    path_gamma=stage_gammas[j],
                    name=f"stage_{i}_attn_{j}",
                )(x)
            elif 1 == j % 3:
                shift_mode = (
                    sum((d + 1) // 3 for d in stage_depths[:i]) + j // 3
                ) % 4 + 1
                x = SwinBlock(
                    stage_dim,
                    stage_window,
                    pretrain_window,
                    num_heads,
                    shift_mode,
                    path_drop=stage_drops[j],
                    expand_ratio=expand_ratio,
                    path_gamma=stage_gammas[j],
                    name=f"stage_{i}_attn_{j}",
                )(x)
            else:
                x = LocalBlock(
                    stage_dim,
                    5,
                    num_heads,
                    dilation_rate=1,
                    kernel_size=3,
                    path_drop=stage_drops[j],
                    expand_ratio=expand_ratio,
                    path_gamma=stage_gammas[j],
                    name=f"stage_{i}_attn_{j}",
                )(x)

            num_heads = stage_dim // 32

        x = layers.Activation("linear", name=f"stage_{i}_out")(x)

    x = Norm(name="norm")(x)

    if include_top:
        if pooling in {None, "avg"}:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif "max" == pooling:
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)
        elif "sp" == pooling:
            x = SimPool(name="sim_pool")(x)
        elif "ma" == pooling:
            x = MultiHeadAttentionPooling(embed_dim // 4, 1, name="ma_pool")(x)
            x = layers.Flatten(name="ma_flat")(x)
        else:
            raise ValueError(
                f"Expecting pooling to be one of None/avg/max/ma. "
                f"Found: {pooling}"
            )

        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, name="head")(x)
        x = layers.Activation(
            classifier_activation, dtype="float32", name="pred"
        )(x)

    if input_tensor is not None:
        inputs = operation_utils.get_source_inputs(input_tensor)
    else:
        inputs = image

    model = models.Model(inputs, x, name=model_name)

    weights_pooling = "avg" if pooling is None else pooling
    weights_top = f"{weights_pooling}_{classes}" if include_top else "notop"
    weights_key = (
        f"{model_name}__{weights_top}__{weights}__{cnapol.global_policy().name}"
    )
    if weights_key in WEIGHT_URLS:
        weights_url = WEIGHT_URLS[weights_key]
        weights_hash = WEIGHT_HASHES[weights_key]
        weights_path = file_utils.get_file(
            origin=weights_url, file_hash=weights_hash, cache_subdir="soft_swin"
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def CoMATiny(
    stem_dim=32,
    stem_depth=2,
    embed_dim=64,
    stage_depths=(3, 6, 21, 3),
    path_drop=0.1,
    pretrain_window=16,
    pretrain_size=256,
    model_name="coma_tiny",
    weights=None,
    classes=14607,
    classifier_activation="linear",
    **kwargs,
):
    # 26.1 13.7
    with cnapol.policy_scope("conv-ln-gelu"):
        return CoMA(
            stem_dim=stem_dim,
            stem_depth=stem_depth,
            embed_dim=embed_dim,
            stage_depths=stage_depths,
            path_drop=path_drop,
            pretrain_window=pretrain_window,
            pretrain_size=pretrain_size,
            model_name=model_name,
            weights=weights,
            classes=classes,
            classifier_activation=classifier_activation,
            **kwargs,
        )


def CoMASmall(
    stem_dim=48,
    stem_depth=2,
    embed_dim=96,
    stage_depths=(3, 6, 21, 3),
    path_drop=0.2,
    pretrain_window=16,
    pretrain_size=256,
    model_name="coma_small",
    weights=None,
    classes=14607,
    classifier_activation="linear",
    **kwargs,
):
    # 56.3 28.0
    with cnapol.policy_scope("conv-ln-gelu"):
        return CoMA(
            stem_dim=stem_dim,
            stem_depth=stem_depth,
            embed_dim=embed_dim,
            stage_depths=stage_depths,
            path_drop=path_drop,
            pretrain_window=pretrain_window,
            pretrain_size=pretrain_size,
            model_name=model_name,
            weights=weights,
            classes=classes,
            classifier_activation=classifier_activation,
            **kwargs,
        )


def CoMABase(
    stem_dim=64,
    stem_depth=3,
    embed_dim=128,
    stage_depths=(3, 6, 21, 3),
    path_drop=0.3,
    current_window=24,
    pretrain_window=12,
    model_name="coma_base",
    weights=None,
    classes=14607,
    classifier_activation="linear",
    **kwargs,
):
    # 98.1 53.6 @ 256
    with cnapol.policy_scope("conv-ln-gelu"):
        return CoMA(
            stem_dim=stem_dim,
            stem_depth=stem_depth,
            embed_dim=embed_dim,
            stage_depths=stage_depths,
            path_drop=path_drop,
            current_window=current_window,
            pretrain_window=pretrain_window,
            model_name=model_name,
            weights=weights,
            classes=classes,
            classifier_activation=classifier_activation,
            **kwargs,
        )


def CoMALarge(
    stem_dim=80,
    stem_depth=4,
    embed_dim=160,
    stage_depths=(6, 9, 27, 3),
    path_drop=0.4,
    current_window=24,
    pretrain_window=12,
    model_name="coma_large",
    weights=None,
    classes=14607,
    classifier_activation="linear",
    **kwargs,
):
    # 172.1 108.5 @ 256
    with cnapol.policy_scope("conv-ln-gelu"):
        return CoMA(
            stem_dim=stem_dim,
            stem_depth=stem_depth,
            embed_dim=embed_dim,
            stage_depths=stage_depths,
            path_drop=path_drop,
            current_window=current_window,
            pretrain_window=pretrain_window,
            model_name=model_name,
            weights=weights,
            classes=classes,
            classifier_activation=classifier_activation,
            **kwargs,
        )

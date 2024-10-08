from functools import partial

import numpy as np
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src.applications import imagenet_utils
from keras.src.applications.efficientnet_v2 import CONV_KERNEL_INITIALIZER
from keras.src.dtype_policies import dtype_policy
from keras.src.ops import operation_utils
from keras.src.utils import file_utils
from keras.src.utils import naming

from segme.common.convnormact import Act
from segme.common.convnormact import Conv
from segme.common.convnormact import Norm
from segme.common.pool import MultiHeadAttentionPooling
from segme.common.pool import SimPool
from segme.policy import cnapol
from segme.policy.backbone.backbone import BACKBONES
from segme.policy.backbone.diy.hardswin import AttnBlock
from segme.policy.backbone.utils import wrap_bone

BASE_URL = (
    "https://github.com/shkarupa-alex/segme/releases/download/3.0.0/"
    "soft_swin_{}__avg_1000__imagenet__conv-ln1em5-gelu___{}.weights.h5"
)
WEIGHT_URLS = {}


def Stem(filters, name=None):
    if name is None:
        counter = naming.get_uid("stem")
        name = f"stem_{counter}"

    def apply(inputs):
        x = Conv(
            filters,
            7,
            strides=2,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=f"{name}_embed",
        )(inputs)
        x = Act(name=f"{name}_act")(x)
        x = Norm(name=f"{name}_norm")(x)

        return x

    return apply


def Reduce(name=None):
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

        x = Conv(
            channels * 2,
            3,
            strides=2,
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=f"{name}_conv",
        )(inputs)
        x = Norm(name=f"{name}_norm")(x)

        return x

    return apply


def SoftSwin(
    embed_dim,
    stage_depths,
    pretrain_window,
    current_window=None,
    expand_ratio=4,
    path_gamma=0.01,
    path_drop=0.2,
    pretrain_size=384,
    current_size=None,
    input_shape=None,
    include_top=True,
    model_name="soft_swin",
    pooling=None,
    weights=None,
    input_tensor=None,
    classes=1000,
    classifier_activation="softmax",
    include_preprocessing=True,
):
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

    path_gammas = np.linspace(path_gamma, 1e-5, sum(stage_depths)).tolist()
    path_drops = np.linspace(0.0, path_drop, sum(stage_depths)).tolist()

    x = Stem(embed_dim // 2, name="stem")(x)
    x = layers.Activation("linear", name="stem_out")(x)

    shift_counter = -1
    for i, stage_depth in enumerate(stage_depths):
        x = Reduce(name=f"stage_{i}_reduce")(x)

        current_stage_window = min(current_window, current_size // 2 ** (i + 2))
        pretrain_stage_window = min(
            pretrain_window, pretrain_size // 2 ** (i + 2)
        )
        num_heads = embed_dim // 2 ** (5 - i)

        stage_gammas, path_gammas = (
            path_gammas[:stage_depth],
            path_gammas[stage_depth:],
        )
        stage_drops, path_drops = (
            path_drops[:stage_depth],
            path_drops[stage_depth:],
        )

        for j in range(stage_depth):
            shift_counter += j % 2
            shift_mode = shift_counter % 4 + 1 if j % 2 else 0
            x = AttnBlock(
                current_stage_window,
                pretrain_stage_window,
                num_heads,
                shift_mode=shift_mode,
                expand_ratio=expand_ratio,
                path_gamma=stage_gammas[j],
                path_drop=stage_drops[j],
                name=f"stage_{i}_attn_{j}",
            )(x)

        x = layers.Activation("linear", name=f"stage_{i}_out")(x)

    x = Norm(name="norm")(x)

    if include_top:
        if pooling in {None, "avg"}:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif "max" == pooling:
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)
        elif "sp" == pooling:
            x = SimPool(embed_dim // 4, name="sim_pool")(x)
        elif "ma" == pooling:
            x = MultiHeadAttentionPooling(
                embed_dim // 4,
                max(1, round(classes / embed_dim / 32)),
                name="ma_pool",
            )(x)
            x = Act(name="ma_act")(x)
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

    model = models.Functional(inputs, x, name=model_name)

    weights_pooling = "avg" if pooling is None else pooling
    weights_top = f"{weights_pooling}_{classes}" if include_top else "notop"
    weights_key = (
        f"{model_name}__{weights_top}__{weights}__{cnapol.global_policy().name}"
    )
    if weights_key in WEIGHT_URLS:
        weights_url = WEIGHT_URLS[weights_key]
        weights_hash = (
            weights_url.split("___")[-1]
            .replace(".weights.h5", "")
            .replace(".h5", "")
        )
        weights_path = file_utils.get_file(
            origin=weights_url, file_hash=weights_hash, cache_subdir="soft_swin"
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def SoftSwinTiny(
    embed_dim=96,
    stage_depths=(2, 2, 6, 2),
    pretrain_window=16,
    pretrain_size=256,
    include_top=False,
    model_name="soft_swin_tiny",
    **kwargs,
):
    with cnapol.policy_scope("conv-ln1em5-gelu"):
        return SoftSwin(
            embed_dim=embed_dim,
            stage_depths=stage_depths,
            pretrain_window=pretrain_window,
            pretrain_size=pretrain_size,
            include_top=include_top,
            model_name=model_name,
            **kwargs,
        )


def SoftSwinSmall(
    embed_dim=96,
    stage_depths=(2, 2, 18, 2),
    pretrain_window=16,
    path_drop=0.3,
    pretrain_size=256,
    include_top=False,
    model_name="soft_swin_small",
    **kwargs,
):
    with cnapol.policy_scope("conv-ln1em5-gelu"):
        return SoftSwin(
            embed_dim=embed_dim,
            stage_depths=stage_depths,
            pretrain_window=pretrain_window,
            path_drop=path_drop,
            pretrain_size=pretrain_size,
            include_top=include_top,
            model_name=model_name,
            **kwargs,
        )


def SoftSwinBase(
    embed_dim=128,
    stage_depths=(2, 2, 18, 2),
    current_window=24,
    pretrain_window=12,
    pretrain_size=192,
    current_size=384,
    include_top=False,
    model_name="soft_swin_base",
    **kwargs,
):
    with cnapol.policy_scope("conv-ln1em5-gelu"):
        return SoftSwin(
            embed_dim=embed_dim,
            stage_depths=stage_depths,
            current_window=current_window,
            pretrain_window=pretrain_window,
            pretrain_size=pretrain_size,
            current_size=current_size,
            include_top=include_top,
            model_name=model_name,
            **kwargs,
        )


def SoftSwinLarge(
    embed_dim=192,
    stage_depths=(2, 2, 18, 2),
    current_window=24,
    pretrain_window=12,
    pretrain_size=192,
    current_size=384,
    include_top=False,
    model_name="soft_swin_large",
    **kwargs,
):
    with cnapol.policy_scope("conv-ln1em5-gelu"):
        return SoftSwin(
            embed_dim=embed_dim,
            stage_depths=stage_depths,
            current_window=current_window,
            pretrain_window=pretrain_window,
            pretrain_size=pretrain_size,
            current_size=current_size,
            include_top=include_top,
            model_name=model_name,
            **kwargs,
        )


BACKBONES.register("softswin_tiny")(
    (
        partial(wrap_bone, SoftSwinTiny, None),
        [
            None,
            "stem_out",
            "stage_0_out",
            "stage_1_out",
            "stage_2_out",
            "stage_3_out",
        ],
    )
)

BACKBONES.register("softswin_small")(
    (
        partial(wrap_bone, SoftSwinSmall, None),
        [
            None,
            "stem_out",
            "stage_0_out",
            "stage_1_out",
            "stage_2_out",
            "stage_3_out",
        ],
    )
)

BACKBONES.register("softswin_base")(
    (
        partial(wrap_bone, SoftSwinBase, None),
        [
            None,
            "stem_out",
            "stage_0_out",
            "stage_1_out",
            "stage_2_out",
            "stage_3_out",
        ],
    )
)

BACKBONES.register("softswin_large")(
    (
        partial(wrap_bone, SoftSwinLarge, None),
        [
            None,
            "stem_out",
            "stage_0_out",
            "stage_1_out",
            "stage_2_out",
            "stage_3_out",
        ],
    )
)

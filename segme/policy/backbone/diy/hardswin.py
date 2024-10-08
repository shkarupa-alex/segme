from functools import partial

import numpy as np
from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import models
from keras.src.applications import imagenet_utils
from keras.src.dtype_policies import dtype_policy
from keras.src.ops import operation_utils
from keras.src.utils import file_utils
from keras.src.utils import naming

from segme.common.attn.swin import SwinAttention
from segme.common.convnormact import Act
from segme.common.convnormact import Conv
from segme.common.convnormact import Norm
from segme.common.drop import DropPath
from segme.common.pool import MultiHeadAttentionPooling
from segme.common.pool import SimPool
from segme.policy import cnapol
from segme.policy.backbone.backbone import BACKBONES
from segme.policy.backbone.utils import wrap_bone

BASE_URL = (
    "https://github.com/shkarupa-alex/segme/releases/download/3.0.0/"
    "hard_swin_{}__avg_1000__imagenet__conv-ln1em5-gelu___{}.weights.h5"
)
WEIGHT_URLS = {
    "hard_swin_tiny__avg_1000__imagenet__conv-ln1em5-gelu": BASE_URL.format(
        "tiny",
        "83d970307e84e328a71482aeed9a6882225ab998d439121df6595b766c8fd1a4",
    ),
    "hard_swin_small__avg_1000__imagenet__conv-ln1em5-gelu": BASE_URL.format(
        "small",
        "a0433c1bc3f68b1faa07f6a9faaeda54ccc9f49341be0e1476bd74199f7cf473",
    ),
    "hard_swin_base__avg_1000__imagenet__conv-ln1em5-gelu": BASE_URL.format(
        "base",
        "2a674cf0f5530a7df967408b84f9d575f0c19c92ad38bf61cba2d063b62b058d",
    ),
    "hard_swin_large__avg_1000__imagenet__conv-ln1em5-gelu": BASE_URL.format(
        "large",
        "eb80bcd8f487171164bc554ac8d32fb8f6b2747b5bf7dab9a6baea6e3f8f4a88",
    ),
}
WEIGHT_HASHES = {
    "hard_swin_tiny__avg_1000__imagenet__conv-ln1em5-gelu": BASE_URL.format(
        "tiny"
    ),
    "hard_swin_small__avg_1000__imagenet__conv-ln1em5-gelu": BASE_URL.format(
        "small"
    ),
    "hard_swin_base__avg_1000__imagenet__conv-ln1em5-gelu": BASE_URL.format(
        "base"
    ),
    "hard_swin_large__avg_1000__imagenet__conv-ln1em5-gelu": BASE_URL.format(
        "large"
    ),
}


def Stem(embed_dim, patch_size, name=None):
    if name is None:
        counter = naming.get_uid("stem")
        name = f"stem_{counter}"

    def apply(inputs):
        x = Conv(
            embed_dim,
            patch_size,
            strides=patch_size,
            padding="valid",
            name=f"{name}_embed",
        )(inputs)
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
            channels * 2, 2, strides=2, use_bias=False, name=f"{name}_conv"
        )(inputs)
        x = Norm(name=f"{name}_norm")(x)

        return x

    return apply


def MLP(expand_ratio=4.0, path_drop=0.0, gamma_initializer="ones", name=None):
    if name is None:
        counter = naming.get_uid("mlp")
        name = f"mlp_{counter}"

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        expand_filters = int(channels * expand_ratio)

        x = layers.Dense(expand_filters, name=f"{name}_expand")(inputs)
        x = Act(name=f"{name}_act")(x)
        x = layers.Dense(channels, name=f"{name}_squeeze")(x)
        x = Norm(gamma_initializer=gamma_initializer, name=f"{name}_norm")(x)
        x = DropPath(path_drop, name=f"{name}_drop")(x)
        x = layers.add([x, inputs], name=f"{name}_add")

        return x

    return apply


def AttnBlock(
    current_window,
    pretrain_window,
    num_heads,
    shift_mode,
    path_drop=0.0,
    expand_ratio=4.0,
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
            shift_mode=shift_mode,
            name=f"{name}_swin_attn",
        )(inputs)
        x = Norm(gamma_initializer=gamma_initializer, name=f"{name}_swin_norm")(
            x
        )
        x = DropPath(path_drop, name=f"{name}_swin_drop")(x)
        x = layers.add([x, inputs], name=f"{name}_swin_add")

        x = MLP(
            expand_ratio=expand_ratio,
            path_drop=path_drop,
            gamma_initializer=gamma_initializer,
            name=f"{name}_mlp",
        )(x)

        return x

    return apply


def HardSwin(
    embed_dim,
    stage_depths,
    pretrain_window,
    current_window=None,
    patch_size=4,
    expand_ratio=4,
    path_gamma=0.01,
    path_drop=0.2,
    pretrain_size=384,
    current_size=None,
    input_shape=None,
    include_top=True,
    model_name="hard_swin",
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

    x = Stem(embed_dim, patch_size, name="stem")(x)
    x = layers.Activation("linear", name="stem_out")(x)

    for i, stage_depth in enumerate(stage_depths):
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
            x = AttnBlock(
                current_stage_window,
                pretrain_stage_window,
                num_heads,
                shift_mode=j % 2,
                expand_ratio=expand_ratio,
                path_gamma=stage_gammas[j],
                path_drop=stage_drops[j],
                name=f"stage_{i}_attn_{j}",
            )(x)

        x = layers.Activation("linear", name=f"stage_{i}_out")(x)
        if i != len(stage_depths) - 1:
            x = Reduce(name=f"stage_{i}_reduce")(x)

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


def HardSwinTiny(
    embed_dim=96,
    stage_depths=(2, 2, 6, 2),
    pretrain_window=16,
    pretrain_size=256,
    include_top=False,
    model_name="hard_swin_tiny",
    **kwargs,
):
    with cnapol.policy_scope("conv-ln1em5-gelu"):
        return HardSwin(
            embed_dim=embed_dim,
            stage_depths=stage_depths,
            pretrain_window=pretrain_window,
            pretrain_size=pretrain_size,
            include_top=include_top,
            model_name=model_name,
            **kwargs,
        )


def HardSwinSmall(
    embed_dim=96,
    stage_depths=(2, 2, 18, 2),
    pretrain_window=16,
    path_drop=0.3,
    pretrain_size=256,
    include_top=False,
    model_name="hard_swin_small",
    **kwargs,
):
    with cnapol.policy_scope("conv-ln1em5-gelu"):
        return HardSwin(
            embed_dim=embed_dim,
            stage_depths=stage_depths,
            pretrain_window=pretrain_window,
            path_drop=path_drop,
            pretrain_size=pretrain_size,
            include_top=include_top,
            model_name=model_name,
            **kwargs,
        )


def HardSwinBase(
    embed_dim=128,
    stage_depths=(2, 2, 18, 2),
    current_window=24,
    pretrain_window=12,
    pretrain_size=192,
    include_top=False,
    current_size=384,
    model_name="hard_swin_base",
    **kwargs,
):
    with cnapol.policy_scope("conv-ln1em5-gelu"):
        return HardSwin(
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


def HardSwinLarge(
    embed_dim=192,
    stage_depths=(2, 2, 18, 2),
    current_window=24,
    pretrain_window=12,
    pretrain_size=192,
    include_top=False,
    current_size=384,
    model_name="hard_swin_large",
    **kwargs,
):
    with cnapol.policy_scope("conv-ln1em5-gelu"):
        return HardSwin(
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


BACKBONES.register("hardswin_tiny")(
    (
        partial(wrap_bone, HardSwinTiny, None),
        [
            None,
            None,
            "stage_0_out",
            "stage_1_out",
            "stage_2_out",
            "stage_3_out",
        ],
    )
)

BACKBONES.register("hardswin_small")(
    (
        partial(wrap_bone, HardSwinSmall, None),
        [
            None,
            None,
            "stage_0_out",
            "stage_1_out",
            "stage_2_out",
            "stage_3_out",
        ],
    )
)

BACKBONES.register("hardswin_base")(
    (
        partial(wrap_bone, HardSwinBase, None),
        [
            None,
            None,
            "stage_0_out",
            "stage_1_out",
            "stage_2_out",
            "stage_3_out",
        ],
    )
)

BACKBONES.register("hardswin_large")(
    (
        partial(wrap_bone, HardSwinLarge, None),
        [
            None,
            None,
            "stage_0_out",
            "stage_1_out",
            "stage_2_out",
            "stage_3_out",
        ],
    )
)

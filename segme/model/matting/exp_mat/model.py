import numpy as np
from keras.src import initializers
from keras.src import layers
from keras.src import models
from keras.src.utils import naming

from segme.common.align import Align
from segme.common.backbone import Backbone
from segme.common.convnormact import Act
from segme.common.convnormact import Conv
from segme.common.convnormact import Norm
from segme.common.drop import DropPath
from segme.common.fold import UnFold
from segme.common.head import HeadProjection
from segme.common.split import Split
from segme.model.matting.exp_mat.trimap import Trimap
from segme.policy import cnapol
from segme.policy import dtpol
from segme.policy.backbone.diy.hardswin import AttnBlock
from segme.policy.backbone.utils import patch_channels


def Encoder():
    image = layers.Input(name="image", shape=(None, None, 3), dtype="uint8")
    trimap = layers.Input(name="trimap", shape=(None, None, 1), dtype="uint8")
    inputs = layers.concatenate(
        [image, Trimap(name="trimap1h")(trimap)],
        axis=-1,
        name="concat",
        dtype="uint8",
    )

    backbone = Backbone(input_tensor=inputs)
    trimap_mean = np.array([0.258, 0.496, 0.247], "float32") * 255.0
    trimap_variance = (np.array([0.437, 0.499, 0.431], "float32") * 255.0) ** 2
    backbone = patch_channels(
        backbone,
        trimap_mean.tolist(),
        trimap_variance.tolist(),
    )

    return backbone


def Attention(
    depth,
    window_size,
    shift_mode,
    expand_ratio=3.0,
    path_drop=0.0,
    path_gamma=1.0,
    name=None,
):
    if name is None:
        counter = naming.get_uid("attn")
        name = f"attn_{counter}"

    if isinstance(path_gamma, float):
        path_gamma = [path_gamma] * depth
    elif len(path_gamma) != depth:
        raise ValueError("Number of path gammas must equals to depth.")

    if isinstance(path_drop, float):
        path_drop = [path_drop] * depth
    elif len(path_drop) != depth:
        raise ValueError("Number of path dropouts must equals to depth.")

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        num_heads = channels // 32

        x = inputs
        for i in range(depth):
            current_shift = (shift_mode + i // 2 - 1) % 4 + 1 if i % 2 else 0
            x = AttnBlock(
                window_size,
                window_size,
                num_heads,
                current_shift,
                path_drop=path_drop[i],
                expand_ratio=expand_ratio,
                path_gamma=path_gamma[i],
                name=f"{name}_{i}",
            )(x)

        return x

    return apply


def FMBConv(
    depth,
    kernel_size=3,
    expand_ratio=3.0,
    path_drop=0.0,
    path_gamma=1.0,
    name=None,
):
    if name is None:
        counter = naming.get_uid("fmbconv")
        name = f"fmbconv_{counter}"

    if isinstance(path_gamma, float):
        path_gamma = [path_gamma] * depth
    elif len(path_gamma) != depth:
        raise ValueError("Number of path gammas must equals to depth.")

    if isinstance(path_drop, float):
        path_drop = [path_drop] * depth
    elif len(path_drop) != depth:
        raise ValueError("Number of path dropouts must equals to depth.")

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        expand_filters = int(channels * expand_ratio)

        x = inputs_ = inputs
        for i in range(depth * 2):
            x = Conv(
                expand_filters,
                kernel_size,
                use_bias=False,
                name=f"{name}_{i}_fmbconv_expand",
            )(inputs_)
            x = Act(name=f"{name}_{i}_act")(x)
            x = Conv(
                channels, 1, use_bias=False, name=f"{name}_{i}_fmbconv_squeeze"
            )(x)
            x = Norm(
                center=False,
                gamma_initializer=initializers.Constant(path_gamma[i // 2]),
                name=f"{name}_{i}_fmbconv_norm",
            )(x)
            x = DropPath(path_drop[i // 2], name=f"{name}_{i}_fmbconv_drop")(x)
            x = layers.add([x, inputs_], name=f"{name}_{i}_fmbconv_add")
            inputs_ = x

        return x

    return apply


def Head(stride, kernel, name=None):
    if name is None:
        counter = naming.get_uid("head")
        name = f"head_{counter}"

    def apply(inputs):
        x = HeadProjection(
            7 * stride**2, kernel_size=kernel, name=f"{name}_logits"
        )(inputs)
        x = UnFold(stride, name=f"{name}_unfold")(x)
        x = layers.Activation(
            "hard_sigmoid", dtype="float32", name=f"{name}_act_a"
        )(x)

        return x

    return apply


def ExpMat(
    transform_depth=2,
    window_size=16,
    path_gamma=0.1,
    path_drop=0.2,
    dtype=None,
):
    if dtype is not None:
        with dtpol.policy_scope(dtype):
            return ExpMat(
                transform_depth=transform_depth,
                window_size=window_size,
                path_gamma=path_gamma,
                path_drop=path_drop,
                dtype=None,
            )

    backbone = Encoder()
    image, trimap = backbone.inputs
    outputs = backbone.outputs[::-1]

    num_shifts = transform_depth // 3 + transform_depth % 3 // 2
    path_size = transform_depth * 2 * (len(outputs) - 1)
    path_gammas = np.linspace(1e-5, path_gamma, path_size).tolist()
    path_drops = np.linspace(path_drop, 0.0, path_size).tolist()

    with cnapol.policy_scope("conv-ln1em5-gelu"):
        heads, o_prev = [], None
        for i, o in enumerate(outputs):
            channels = o.shape[-1]
            if channels is None:
                raise ValueError(
                    "Channel dimension of the inputs should be defined. "
                    "Found `None`."
                )

            stride = 2 ** (5 - i)

            o = Conv(channels, 1, name=f"backstage_{i}_lateral_proj")(o)
            o = Act(name=f"backstage_{i}_lateral_act")(o)
            o = Norm(name=f"backstage_{i}_lateral_norm")(o)

            if o_prev is None:
                o_prev = o
                heads.append(Head(stride, 1, name=f"head_{i}")(o))
                continue

            shift_mode = num_shifts * (i - 1) * 2 % 4 + 1
            stage_drops, path_drops = (
                path_drops[:transform_depth],
                path_drops[transform_depth:],
            )
            stage_gammas, path_gammas = (
                path_gammas[:transform_depth],
                path_gammas[transform_depth:],
            )
            if i < 4:
                o = Attention(
                    transform_depth,
                    window_size,
                    shift_mode,
                    path_drop=stage_drops,
                    path_gamma=stage_gammas,
                    name=f"backstage_{i}_lateral_transform",
                )(o)
            else:
                o = FMBConv(
                    transform_depth,
                    path_drop=stage_drops,
                    path_gamma=stage_gammas,
                    name=f"backstage_{i}_lateral_transform",
                )(o)

            o = Align(channels, name=f"backstage_{i}_merge_align")([o, o_prev])
            o = Norm(name=f"backstage_{i}_merge_norm")(o)

            shift_mode = (num_shifts * (i - 1) * 2 + num_shifts) % 4 + 1
            stage_drops, path_drops = (
                path_drops[:transform_depth],
                path_drops[transform_depth:],
            )
            stage_gammas, path_gammas = (
                path_gammas[:transform_depth],
                path_gammas[transform_depth:],
            )
            if i < 4:
                o = Attention(
                    transform_depth,
                    window_size,
                    shift_mode,
                    path_drop=stage_drops,
                    path_gamma=stage_gammas,
                    name=f"backstage_{i}_merge_transform",
                )(o)
            else:
                o = FMBConv(
                    transform_depth,
                    path_drop=stage_drops,
                    path_gamma=stage_gammas,
                    name=f"backstage_{i}_merge_transform",
                )(o)

            o_prev = o
            heads.append(Head(stride, 3, name=f"head_{i}")(o))

        a, f, b = Split([1, 4], name="a_split", dtype="float32")(heads[-1])
        heads.extend([a, f, b])

        model = models.Functional(
            inputs={"image": image, "trimap": trimap},
            outputs=tuple(heads),
            name="exp_mat",
        )

        return model

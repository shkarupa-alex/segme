import numpy as np
from keras.src import layers
from keras.src import models
from keras.src.utils import file_utils
from keras.src.utils import naming

from segme.common.convnormact import Act
from segme.common.convnormact import Conv
from segme.common.convnormact import Norm
from segme.common.head import ClassificationActivation
from segme.common.resize import NearestInterpolation
from segme.policy import cnapol
from segme.policy import dtpol

BASE_URL = (
    "https://github.com/shkarupa-alex/segme/releases/download/3.0.0/"
    "seg_refiner__{}__{}___{}.weights.h5"
)
WEIGHT_URLS = {
    "seg_refiner__lr__256": BASE_URL.format(
        "lr",
        256,
        "3f4bd017f52a6073e2b3e7674b9778f7196a30d56b90915a3019cf7a9e7b7701",
    ),
    "seg_refiner__hr__256": BASE_URL.format(
        "hr",
        256,
        "17ca665293276cecaae83f9bd8d28942929dfc42dc46c961ebfb37c91f22f5d3",
    ),
}


def _ResBlock(filters, dropout, name=None):
    if name is None:
        counter = naming.get_uid("res_block")
        name = f"res_block_{counter}"

    def apply(inputs, time_embeds):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        time_embeds = Act(name=f"{name}_time_act")(time_embeds)
        time_embeds = layers.Dense(filters, name=f"{name}_time_proj")(
            time_embeds
        )

        x = Norm(name=f"{name}_in_norm")(inputs)
        x = Act(name=f"{name}_in_act")(x)
        x = Conv(filters, 3, name=f"{name}_in_conv")(x)

        x = layers.add([x, time_embeds], name=f"{name}_time_add")

        x = Norm(name=f"{name}_out_norm")(x)
        x = Act(name=f"{name}_out_act")(x)
        x = layers.Dropout(dropout, name=f"{name}_out_drop")(x)
        x = Conv(filters, 3, name=f"{name}_out_conv")(x)

        if filters == channels:
            skip = inputs
        else:
            skip = Conv(filters, 1, name=f"{name}_skip_proj")(inputs)

        x = layers.add([skip, x], name=f"{name}_skip_add")

        return x

    return apply


def SegRefiner(
    size=256,
    filters=128,
    depth=2,
    atstrides=(16, 32),
    dropout=0,
    mults=(1, 1, 2, 2, 4, 4),
    heads=4,
    weights=None,
    dtype=None,
):
    if dtype is not None:
        with dtpol.policy_scope(dtype):
            return SegRefiner(
                filters=filters,
                depth=depth,
                atstrides=atstrides,
                dropout=dropout,
                mults=mults,
                heads=heads,
                weights=weights,
                dtype=None,
            )

    with cnapol.policy_scope("conv-gn321em5-silu"):
        image = layers.Input(name="image", shape=(size, size, 3), dtype="uint8")
        mask = layers.Input(name="mask", shape=(size, size, 1), dtype="uint8")
        time = layers.Input(name="time", shape=[], dtype="int32")

        combo = layers.concatenate(
            [
                layers.Normalization(
                    mean=np.array([0.485, 0.456, 0.406], "float32") * 255.0,
                    variance=(
                        np.array([0.229, 0.224, 0.225], "float32") * 255.0
                    )
                    ** 2,
                    name="image_normalize",
                )(image),
                layers.Rescaling(1 / 255, name="mask_rescale")(mask),
            ],
            name="inputs_concat",
        )

        time_embed = layers.Embedding(6, filters * 4, name="time_embed")(time)

        x = Conv(filters, 3, name="in_0_0_conv")(combo)
        skip_connections = [x]

        index = 1
        for i, mult in enumerate(mults):
            for j in range(depth):
                x = _ResBlock(mult * filters, dropout, name=f"in_{index}_0")(
                    x, time_embed
                )

                if 2**i in atstrides:
                    y = Norm(name=f"in_{index}_1_norm")(x)
                    y = layers.MultiHeadAttention(
                        heads,
                        mult * filters // heads,
                        name=f"in_{index}_1_attn",
                    )(y, y)
                    x = layers.add([x, y], name=f"in_{index}_1_skip_add")

                skip_connections.append(x)
                index += 1

            if i != len(mults) - 1:
                x = Conv(
                    mult * filters,
                    3,
                    strides=2,
                    name=f"in_{index}_2_down",
                )(x)

                skip_connections.append(x)
                index += 1

        x = _ResBlock(mults[-1] * filters, dropout, name="mid_0")(x, time_embed)

        y = Norm(name="mid_1_norm")(x)
        y = layers.MultiHeadAttention(
            heads, mults[-1] * filters // heads, name="mid_1_attn"
        )(y, y)
        x = layers.add([x, y], name="mid_1_skip_add")

        x = _ResBlock(mults[-1] * filters, dropout, name="mid_2")(x, time_embed)

        index = 0
        for i, mult in list(enumerate(mults))[::-1]:
            for j in range(depth + 1):
                x = layers.concatenate(
                    [x, skip_connections.pop()],
                    name=f"out_{index}_concat",
                )

                x = _ResBlock(filters * mult, dropout, name=f"out_{index}_0")(
                    x, time_embed
                )

                if 2**i in atstrides:
                    y = Norm(name=f"out_{index}_1_norm")(x)
                    y = layers.MultiHeadAttention(
                        heads,
                        mult * filters // heads,
                        name=f"out_{index}_1_attn",
                    )(y, y)
                    x = layers.add([x, y], name=f"out_{index}_1_skip_add")

                if i and j == depth:
                    x = NearestInterpolation(2, name=f"out_{index}_2_expand")(x)
                    x = Conv(mult * filters, 3, name=f"out_{index}_2_up")(x)

                index += 1

        x = Norm(name="pred_norm")(x)
        x = Act(name="pred_act")(x)
        x = Conv(1, 3, name="pred_proj")(x)

        x = ClassificationActivation(name="pred_prob")(x)

        model = models.Functional(
            inputs=[image, mask, time], outputs=x, name="seg_refiner"
        )

        if weights in WEIGHT_URLS:
            weights_url = WEIGHT_URLS[weights]
            weights_hash = (
                weights_url.split("___")[-1]
                .replace(".weights.h5", "")
                .replace(".h5", "")
            )
            weights_path = file_utils.get_file(
                origin=weights_url,
                file_hash=weights_hash,
                cache_subdir="seg_refiner",
            )
            model.load_weights(weights_path)
        elif weights is not None:
            model.load_weights(weights)

        return model

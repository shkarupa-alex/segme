from keras.src import layers
from keras.src import models

from segme.common.convnormact import ConvNormAct
from segme.common.head import ClassificationActivation
from segme.common.onem import OneMinus
from segme.common.resize import BilinearInterpolation
from segme.common.sequence import Sequence
from segme.policy import dtpol


def HierarchicalMultiScaleAttention(
    model, features, logits, scales, filters=256, dropout=0.0, dtype=None
):
    if dtype is not None:
        with dtpol.policy_scope(dtype):
            return HierarchicalMultiScaleAttention(
                model=model,
                features=features,
                logits=logits,
                scales=scales,
                filters=filters,
                dropout=dropout,
                dtype=None,
            )

    # Use scales (0.5,) for training and (0.25, 0.5, 2.0) for inference
    if not isinstance(model, models.Functional):
        raise ValueError(
            f"Expecting model to be an instance of `keras.models.Functional`. "
            f"Got: {type(model)}"
        )

    if 1 != len(model.inputs):
        raise ValueError("Models with multiple inputs not supported.")

    scales = sorted({1.0} | set(scales), reverse=True)
    if len(scales) < 2:
        raise ValueError(
            "Expecting `scales` to have at least one more scale except `1`."
        )

    model_ = models.Functional(
        inputs=model.inputs,
        outputs=(
            model.get_layer(name=features).output,
            model.get_layer(name=logits).output,
        ),
    )
    attention = Sequence(
        [
            ConvNormAct(filters, 3, name="attention_cna1"),
            ConvNormAct(filters, 3, name="attention_cna2"),
            layers.Dropout(dropout, name="_drop"),
            layers.Conv2D(
                1,
                1,
                activation="sigmoid",
                use_bias=False,
                name="attention_proj",
            ),
        ],
        name="attention",
    )

    inputs = layers.Input(
        name="image",
        shape=model_.inputs[0].shape[1:],
        dtype=model_.inputs[0].dtype,
    )

    logits = None
    for i, scale in enumerate(scales):
        inputs_ = BilinearInterpolation(scale, name=f"resize_inputs_{i}")(
            inputs
        )
        features_, logits_ = model_(inputs_)

        if logits is None:
            logits = logits_
            continue

        attention_ = attention(features_)

        if scale >= 1.0:
            # downscale previous
            attention_ = BilinearInterpolation(name=f"resize_attention_{i}")(
                [attention_, logits_]
            )
            logits = BilinearInterpolation(name=f"resize_outputs_{i}")(
                [logits, logits_]
            )
        elif scale < 1.0:
            # upscale current
            attention_ = BilinearInterpolation(name=f"resize_attention_{i}")(
                [attention_, logits]
            )
            logits_ = BilinearInterpolation(name=f"resize_outputs{i}_left")(
                [logits_, logits]
            )

        logits = layers.add(
            [
                layers.multiply(
                    [
                        OneMinus(name=f"attention_{i}_invert")(attention_),
                        logits,
                    ],
                    name=f"attention_{i}_prev",
                ),
                layers.multiply(
                    [attention_, logits_], name=f"attention_{i}_curr"
                ),
            ],
            name=f"attention_{i}_add",
        )

    probs = ClassificationActivation(name="act")(logits)

    model_ = models.Functional(inputs=inputs, outputs=probs)

    return model_

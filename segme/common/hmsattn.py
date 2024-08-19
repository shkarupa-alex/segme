import keras
from keras.src import layers
from keras.src import models

from segme.common.convnormact import ConvNormAct
from segme.common.resize import BilinearInterpolation
from segme.common.sequence import Sequence


def HierarchicalMultiScaleAttention(
    model, features, logits, scales, filters=256, dropout=0.0
):
    if not isinstance(model, keras.Model):
        raise ValueError(
            f"Expecting teacher model to be an instance of `keras.Model`. "
            f"Got: {type(model)}"
        )

    scales = sorted({1.0} | set(scales), reverse=True)
    if len(scales) < 2:
        raise ValueError(
            "Expecting `scales` to have at least one more scale except `1`."
        )

    model = models.Model(
        inputs=model.inputs[0],
        outputs=(
            model.get_layer(name=features).output,
            model.get_layer(name=logits).output,
        ),
    )

    inputs = layers.Input(name="image", shape=(None, None, 3), dtype="uint8")

    outputs = None
    for i, scale in enumerate(scales):
        _inputs = BilinearInterpolation(scale, name=f"resize_inputs{i}")(inputs)
        _features, _logits = model(_inputs)

        if outputs is None:
            outputs = _logits
            continue

        _attention = Sequence(
            [
                ConvNormAct(filters, 3, name=f"attention{i}_cna1"),
                ConvNormAct(filters, 3, name=f"attention{i}_cna2"),
                layers.Dropout(dropout, name=f"attention{i}_drop"),
                layers.Conv2D(
                    1,
                    1,
                    activation="sigmoid",
                    use_bias=False,
                    name=f"attention{i}_proj",
                ),
            ],
            name=f"attention{i}",
        )(_features)
        _attention = BilinearInterpolation(None, name=f"resize_attention{i}")(
            [_attention, _inputs]
        )

        if scale >= 1.0:
            # downscale previous
            outputs = BilinearInterpolation(None, name=f"resize_outputs{i}")(
                [outputs, _logits]
            )
            outputs = layers.add(
                [
                    layers.multiply([_attention, _logits])
                    + layers.multiply(
                        [layers.subtract([1.0, _attention]), outputs]
                    )
                ]
            )
        else:
            # upscale current
            _logits = layers.multiply(
                [_attention, _logits], name=f"multiply_outputs{i}_left"
            )
            _logits = BilinearInterpolation(
                None, name=f"resize_outputs{i}_left"
            )([_logits, outputs])
            _attention = BilinearInterpolation(
                None, name=f"resize_outputs{i}_right"
            )([_attention, outputs])

            outputs = layers.add(
                [
                    _logits,
                    layers.multiply(
                        [layers.subtract([1.0, _attention]), outputs]
                    ),
                ]
            )

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

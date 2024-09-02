from keras.src import layers
from keras.src import models

from segme.common.backbone import Backbone
from segme.common.convnormact import ConvNormAct
from segme.common.head import ClassificationActivation
from segme.common.head import HeadProjection
from segme.common.ppm import PyramidPooling
from segme.common.resize import BilinearInterpolation
from segme.policy import dtpol


def UPerNet(classes, decoder_filters=256, head_dropout=0.1, dtype=None):
    if dtype is not None:
        with dtpol.policy_scope(dtype):
            return UPerNet(
                classes,
                decoder_filters=decoder_filters,
                head_dropout=head_dropout,
                dtype=None,
            )

    backbone = Backbone()

    inputs = backbone.inputs[0]
    features = backbone.outputs
    scales = len(features)

    laterals = [
        ConvNormAct(decoder_filters, 1, name=f"lateral_{i}")(features[i])
        for i in range(scales - 1)
    ]
    laterals.append(PyramidPooling(decoder_filters, name="ppm")(features[-1]))

    for i in range(scales - 1, 0, -1):
        lat_up = BilinearInterpolation(name=f"upscale_{i}")(
            [laterals[i], laterals[i - 1]]
        )
        laterals[i - 1] = layers.add([laterals[i - 1], lat_up])

    outputs = [
        ConvNormAct(decoder_filters, 3, name=f"fpn_{i}")(laterals[i])
        for i in range(scales - 1)
    ]
    outputs.append(laterals[-1])

    for i in range(scales - 1, 0, -1):
        outputs[i] = BilinearInterpolation(name=f"resize_{i}")(
            [outputs[i], outputs[0]]
        )

    outputs = layers.concatenate(outputs, axis=-1)
    outputs = ConvNormAct(decoder_filters, 3, name="bottleneck")(outputs)

    outputs = layers.Dropout(head_dropout, name="head_drop")(outputs)
    outputs = HeadProjection(classes, name="head_proj")(outputs)
    outputs = BilinearInterpolation(name="head_resize")([outputs, inputs])
    outputs = ClassificationActivation(name="head_act")(outputs)

    model = models.Model(inputs=inputs, outputs=outputs, name="uper_net")

    return model

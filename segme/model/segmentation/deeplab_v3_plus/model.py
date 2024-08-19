from keras.src import Model
from keras.src import layers

from segme.common.aspp import AtrousSpatialPyramidPooling
from segme.common.backbone import Backbone
from segme.common.convnormact import ConvNormAct
from segme.common.head import ClassificationActivation
from segme.common.head import HeadProjection
from segme.common.hmsattn import HierarchicalMultiScaleAttention
from segme.common.resize import BilinearInterpolation
from segme.common.sequence import Sequence
from segme.policy import dtpol


def DeepLabV3Plus(
    classes,
    aspp_filters=256,
    aspp_stride=32,
    low_filters=48,
    decoder_filters=256,
    dtype=None,
):
    if dtype is not None:
        with dtpol.policy_scope(dtype):
            return DeepLabV3Plus(
                classes,
                aspp_filters=aspp_filters,
                aspp_stride=aspp_stride,
                low_filters=low_filters,
                decoder_filters=decoder_filters,
                dtype=None,
            )

    inputs = layers.Input(name="image", shape=(None, None, 3), dtype="uint8")

    features = Backbone([4, 32])(inputs)

    fine, coarse = features[-2:]
    fine = ConvNormAct(low_filters, 1, name="decoder_fine_proj")(fine)
    coarse = AtrousSpatialPyramidPooling(
        aspp_filters, aspp_stride, name="decoder_coarse_aspp"
    )(coarse)
    coarse = BilinearInterpolation(None, name="decoder_coarse_resize")(
        [coarse, fine]
    )

    outputs = layers.concatenate([fine, coarse], axis=-1, name="decoder_concat")
    outputs = Sequence(
        [
            ConvNormAct(None, 3, name="decoder_proj1_dw"),
            ConvNormAct(decoder_filters, 1, name="decoder_proj1_pw"),
            ConvNormAct(None, 3, name="decoder_proj2_dw"),
            ConvNormAct(decoder_filters, 1, name="decoder_proj2_pw"),
        ],
        name="decoder_proj",
    )(outputs)

    outputs = HeadProjection(classes, name="logits_proj")(outputs)

    outputs = BilinearInterpolation(None, name="logits_resize")(
        [outputs, inputs]
    )
    outputs = ClassificationActivation(name="probs")(outputs)

    model = Model(inputs=inputs, outputs=outputs, name="deeplab_v3_plus")

    return model


def DeepLabV3PlusHMS(
    classes,
    aspp_filters=256,
    aspp_stride=32,
    low_filters=48,
    decoder_filters=256,
    scales=(0.5,),
    dtype=None,
):
    # Rebuild with scales = (0.25, 0.5, 2.0) for inference

    if dtype is not None:
        with dtpol.policy_scope(dtype):
            return DeepLabV3PlusHMS(
                classes,
                aspp_filters=aspp_filters,
                aspp_stride=aspp_stride,
                low_filters=low_filters,
                decoder_filters=decoder_filters,
                scales=scales,
                dtype=None,
            )

    return HierarchicalMultiScaleAttention(
        DeepLabV3Plus(
            classes,
            aspp_filters=aspp_filters,
            aspp_stride=aspp_stride,
            low_filters=low_filters,
            decoder_filters=decoder_filters,
            dtype=dtype,
        ),
        "decoder_proj",
        "logits_resize",
        scales=scales,
        filters=256,
        dropout=0.0,
    )

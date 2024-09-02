from keras.src import layers
from keras.src import models
from keras.src.utils import naming

from segme.common.align import Align
from segme.common.backbone import Backbone
from segme.common.convnormact import ConvNormAct
from segme.common.fold import UnFold
from segme.common.head import ClassificationActivation
from segme.common.head import HeadProjection
from segme.common.pool import AdaptiveAveragePooling
from segme.common.resize import BilinearInterpolation
from segme.policy.backbone.utils import patch_channels
from segme.policy import dtpol


def Encoder():
    inputs = layers.concatenate([
        layers.Input(name="image", shape=(None, None, 3), dtype="uint8"),
        layers.Input(name="mask", shape=(None, None, 1), dtype="uint8"),
    ], name="concat")
    # TODO: backbone type
    backbone = Backbone([2, 4, 32], inputs, "resnet_rs_50_s8-imagenet")
    backbone = patch_channels(backbone, [0.449], [0.497**2])

    return backbone


def FPP(kernel_size=3, name=None):
    if name is None:
        counter = naming.get_uid("lmpp")
        name = f"lmpp_{counter}"

    def apply(inputs):
        channels = inputs.shape[-1]
        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        x = [ConvNormAct(channels, kernel_size, name=f"{name}_cna")(inputs)]

        atr_channels = (channels // 2 // 8) * 8
        for atr_rate in [12, 24, 36]:
            y = ConvNormAct(
                None,
                kernel_size,
                dilation_rate=atr_rate,
                name=f"{name}_atr{atr_rate}_dna",
            )(inputs)
            y = ConvNormAct(atr_channels, 1, name=f"{name}_atr{atr_rate}_pna")(
                y
            )
            x.append(y)

        avg_channels = (channels // 4 // 8) * 8
        for avg_rate in [1, 2, 3, 6]:
            y = AdaptiveAveragePooling(
                avg_rate, name=f"{name}_avg{avg_rate}_pool"
            )(inputs)
            y = ConvNormAct(avg_channels, 1, name=f"{name}_avg{avg_rate}_pna")(
                y
            )
            y = BilinearInterpolation(name=f"{name}_avg{avg_rate}_up")(
                [y, inputs]
            )
            x.append(y)

        x = layers.concatenate(x, name=f"{name}_merge")
        x = ConvNormAct(channels, 1, name=f"{name}_proj")(x)

        return x

    return apply


def Head(unfold, stride, name=None):
    if name is None:
        counter = naming.get_uid("head")
        name = f"head_{counter}"

    def apply(inputs):
        if unfold:
            x = HeadProjection(stride**2, name=f"{name}_logits")(inputs)
            x = UnFold(stride, name=f"{name}_unfold")(x)
        else:
            x = HeadProjection(1, name=f"{name}_logits")(inputs)
            x = BilinearInterpolation(stride, name=f"{name}_resize")(x)

        x = ClassificationActivation(name=f"{name}_act")(x)

        return x

    return apply


def ExpRef(sup_unfold=False, dtype=None):
    if dtype is not None:
        with dtpol.policy_scope(dtype):
            return ExpRef(
                sup_unfold=sup_unfold,
                dtype=None,
            )

    encoder = Encoder()
    feats2, feats4, feats8 = encoder.outputs

    outputs8 = FPP(name="fpp")(feats8)
    probs8 = Head(sup_unfold, 8, name="head8")(outputs8)

    outputs4 = Align(feats4.shape[-1], name="merge4")([feats4, outputs8])
    probs4 = Head(sup_unfold, 4, name="head4")(outputs4)

    outputs2 = Align(feats2.shape[-1], name="merge2")([feats2, outputs4])
    probs2 = Head(sup_unfold, 2, name="head2")(outputs2)

    model = models.Model(
        inputs=encoder.inputs, outputs=(probs2, probs4, probs8), name="exp_sod"
    )

    return model

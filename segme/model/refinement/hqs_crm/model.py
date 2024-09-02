from keras.src import layers
from keras.src import models
from keras.src.dtype_policies import dtype_policy

from segme.common.backbone import Backbone
from segme.common.convnormact import ConvNormAct
from segme.common.head import ClassificationActivation
from segme.common.resize import BilinearInterpolation
from segme.model.refinement.hqs_crm.decoder import Decoder
from segme.policy import dtpol
from segme.policy.backbone.utils import patch_channels


def HqsCrm(
    aspp_filters=(64, 64, 128),
    aspp_drop=0.5,  # TODO
    mlp_units=(32, 32, 32, 32),
    dtype=None,
):
    if dtype is not None:
        with dtpol.policy_scope(dtype):
            return HqsCrm(
                aspp_filters=aspp_filters,
                aspp_drop=aspp_drop,
                mlp_units=mlp_units,
                dtype=None,
            )

    inputs = layers.concatenate(
        [
            layers.Input(name="image", shape=(None, None, 3), dtype="uint8"),
            layers.Input(name="mask", shape=(None, None, 1), dtype="uint8"),
        ],
        axis=-1,
    )
    coord = layers.Input(
        name="coord",
        shape=[None, None, 2],
        dtype=dtype_policy.dtype_policy().compute_dtype,
    )

    backbone = Backbone(
        [2, 4, 32], input_tensor=inputs, policy="resnet_rs_50_s8-imagenet"
    )
    backbone = patch_channels(backbone, mean=[0.408], variance=[0.492**2])
    feats2, feats4, feats8 = backbone.outputs

    # TODO: kernel=1?
    aspp2 = ConvNormAct(aspp_filters[0], 1, name="cna2")(feats2)
    aspp2 = layers.Dropout(aspp_drop, name="drop2")(aspp2)

    aspp4 = ConvNormAct(aspp_filters[1], 1, name="cna4")(feats4)
    aspp4 = layers.Dropout(aspp_drop, name="drop4")(aspp4)
    aspp4 = BilinearInterpolation(name="resize4")([aspp4, aspp2])

    aspp8 = ConvNormAct(aspp_filters[2], 1, name="cna8")(feats8)
    aspp8 = layers.Dropout(aspp_drop, name="drop8")(aspp8)
    aspp8 = BilinearInterpolation(name="resize8")([aspp8, aspp2])

    aspp = layers.concatenate([aspp2, aspp4, aspp8], axis=-1, name="concat")
    aspp = ConvNormAct(sum(aspp_filters), 3, name="fuse")(aspp)
    aspp = layers.Dropout(aspp_drop, name="drop")(aspp)

    logits = Decoder(mlp_units, name="query")([aspp, coord])

    x = ClassificationActivation(name="prob")(logits)

    model = models.Model(
        inputs=backbone.inputs + [coord], outputs=x, name="hqs_crm"
    )

    return model

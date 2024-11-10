from keras.src import layers
from keras.src import models
from keras.src.utils import file_utils

from segme.common.backbone import Backbone
from segme.common.convnormact import Act
from segme.common.convnormact import ConvNormAct
from segme.common.head import HeadProjection
from segme.common.ppm import PyramidPooling
from segme.common.resize import BilinearInterpolation
from segme.common.sequence import Sequence
from segme.model.matting.fba_matting.fusion import Fusion
from segme.policy import cnapol
from segme.policy import dtpol
from segme.policy.backbone.utils import patch_channels

WEIGHT_URLS = {
    "nostd_noexcl_norunaug_512": "https://github.com/shkarupa-alex/segme/releases/download/3.0.0/"
    "fba_matting__nostd_noexcl_norunaug_512___5d202ea93a71ca5e5b23c980950"
    "e3b05bb230fa7c933cd17f498f2538b39350d.weights.h5",
}


def Encoder():
    image = layers.Input(name="image", shape=(None, None, 3), dtype="uint8")
    twomap = layers.Input(name="twomap", shape=[None, None, 2], dtype="uint8")
    distance = layers.Input(
        name="distance", shape=[None, None, 6], dtype="uint8"
    )

    # Rescale twomap and distance to match preprocessed image
    inputs = layers.concatenate(
        [image, twomap, distance], axis=-1, name="concat"
    )
    backbone = Backbone([2, 4, 32], inputs, "resnet_rs_50_s8-imagenet")
    backbone = patch_channels(
        backbone,
        [0.306, 0.311, 0.331, 0.402, 0.485, 0.340, 0.417, 0.498],
        [
            0.461**2,
            0.463**2,
            0.463**2,
            0.462**2,
            0.450**2,
            0.465**2,
            0.464**2,
            0.452**2,
        ],
    )

    return backbone


def FBAMatting(weights="nostd_noexcl_norunaug_512", dtype=None):
    if dtype is not None:
        with dtpol.policy_scope(dtype):
            return FBAMatting(dtype=None)

    with cnapol.policy_scope("conv-gn321em5-leakyrelu"):
        backbone = Encoder()

        image, twomap, _ = backbone.inputs
        feats2, feats4, feats8 = backbone.outputs

        imscal = layers.Rescaling(1 / 255, name="image_scale")(image)
        imnorm = layers.Normalization(
            mean=[0.485, 0.456, 0.406],
            variance=[0.229**2, 0.224**2, 0.225**2],
            name="image_normalize",
        )(imscal)
        tmscal = layers.Rescaling(1 / 255, name="twomap_scale")(twomap)

        x = PyramidPooling(256, name="pyramid_pool")(feats8)
        x = ConvNormAct(256, 3, name="pyramid_cna")(x)

        x = BilinearInterpolation(name="resize_8")([x, feats4])
        x = layers.concatenate([x, feats4], axis=-1, name="concat_8")
        x = ConvNormAct(256, 3, name="cna_8")(x)

        x = BilinearInterpolation(name="resize_4")([x, feats2])
        x = layers.concatenate([x, feats2], axis=-1, name="concat_4")
        x = ConvNormAct(64, 3, name="cna_4")(x)

        x = BilinearInterpolation(name="resize_2")([x, imscal])
        x = layers.concatenate(
            [x, imscal, imnorm, tmscal], axis=-1, name="concat_2"
        )
        x = Sequence(
            [
                layers.Conv2D(32, 3, padding="same", name="proj_conv_0"),
                Act(name="proj_act_0"),
                layers.Conv2D(16, 3, padding="same", name="proj_conv_1"),
                Act(name="proj_act_1"),
                HeadProjection(7, name="proj_head"),
            ],
            name="proj",
        )(x)

        alfgbg, alpha, foreground, background = Fusion(name="fuse")([imscal, x])

        model = models.Functional(
            inputs={i.name: i for i in backbone.inputs},
            outputs=(alfgbg, alpha, foreground, background),
            name="fba_matting",
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

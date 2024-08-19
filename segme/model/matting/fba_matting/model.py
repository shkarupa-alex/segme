from keras.src import layers
from keras.src import models

from segme.common.convnormact import Act
from segme.common.convnormact import ConvNormAct
from segme.common.head import HeadProjection
from segme.common.ppm import PyramidPooling
from segme.common.resize import BilinearInterpolation
from segme.common.sequence import Sequence
from segme.model.matting.fba_matting.fusion import Fusion
from segme.policy import bbpol
from segme.policy import cnapol
from segme.policy import dtpol
from segme.policy.backbone.utils import patch_config


def Encoder():
    base_model = bbpol.BACKBONES.new(
        "resnet_rs_50_s8", "imagenet", 3, [2, 4, 32]
    )
    base_model.trainable = True

    base_config = base_model.get_config()
    base_weights = base_model.get_weights()

    ext_config = base_config
    ext_config["layers"][0]["config"]["batch_shape"] = (None, None, None, 11)
    ext_config["layers"][1]["build_config"]["input_shape"] = (
        None,
        None,
        None,
        11,
    )
    ext_config["layers"][1]["inbound_nodes"][0]["args"][0]["config"][
        "shape"
    ] = (None, None, None, 11)
    ext_config["layers"][2]["build_config"]["input_shape"] = (
        None,
        None,
        None,
        11,
    )
    ext_config["layers"][2]["inbound_nodes"][0]["args"][0]["config"][
        "shape"
    ] = (None, None, None, 11)
    ext_config["layers"][3]["build_config"]["input_shape"] = (
        None,
        None,
        None,
        11,
    )
    ext_config["layers"][3]["inbound_nodes"][0]["args"][0]["config"][
        "shape"
    ] = (None, None, None, 11)
    ext_config = patch_config(
        ext_config,
        [2],
        "mean",
        lambda old: old
        + [0.306, 0.311, 0.331, 0.402, 0.485, 0.340, 0.417, 0.498],
    )
    ext_config = patch_config(
        ext_config,
        [2],
        "variance",
        lambda old: old
        + [
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
    ext_model = models.Model.from_config(ext_config)

    ext_weights = []
    for base_weight, ext_weight in zip(base_weights, ext_model.get_weights()):
        if base_weight.shape != ext_weight.shape:
            if (
                base_weight.shape[:2] + base_weight.shape[3:]
                != ext_weight.shape[:2] + ext_weight.shape[3:]
            ):
                raise ValueError("Unexpected weight shape")

            ext_weight[:, :, : base_weight.shape[2]] = base_weight
            ext_weights.append(ext_weight)
        else:
            ext_weights.append(base_weight)

    ext_model.set_weights(ext_weights)
    ext_model.trainable = True

    return ext_model


def Decoder():
    def apply(feats2, feats4, feats32, imscal, imnorm, twomap):
        x = PyramidPooling(256)(feats32)
        x = ConvNormAct(256, 3)(x)

        x = BilinearInterpolation(None)([x, feats4])
        x = layers.concatenate([x, feats4], axis=-1)
        x = ConvNormAct(256, 3)(x)

        x = BilinearInterpolation(None)([x, feats2])
        x = layers.concatenate([x, feats2], axis=-1)
        x = ConvNormAct(64, 3)(x)

        x = BilinearInterpolation(None)([x, imscal])
        x = layers.concatenate([x, imscal, imnorm, twomap], axis=-1)
        x = Sequence(
            [
                layers.Conv2D(32, 3, padding="same"),
                Act(),
                layers.Conv2D(16, 3, padding="same"),
                Act(),
                HeadProjection(7),
            ]
        )(x)

        return x

    return apply


def FBAMatting(dtype=None):
    if dtype is not None:
        with dtpol.policy_scope(dtype):
            return FBAMatting(dtype=None)

    with cnapol.policy_scope("stdconv-gn-leakyrelu"):
        image = layers.Input(name="image", shape=(None, None, 3), dtype="uint8")
        twomap = layers.Input(
            name="twomap", shape=[None, None, 2], dtype="uint8"
        )
        distance = layers.Input(
            name="distance", shape=[None, None, 6], dtype="uint8"
        )

        # Rescale twomap and distance to match preprocessed image
        featraw = layers.concatenate([image, twomap, distance], axis=-1)
        feats2, feats4, feats32 = Encoder()(featraw)

        imscal = layers.Rescaling(1 / 255)(image)
        imnorm = layers.Normalization(
            mean=[0.485, 0.456, 0.406], variance=[0.229**2, 0.224**2, 0.225**2]
        )(imscal)
        alfgbg = Decoder()(
            feats2,
            feats4,
            feats32,
            imscal,  # scaled image
            imnorm,  # normalized image
            layers.Rescaling(1 / 255)(twomap),  # scaled twomap
        )

        alfgbg, alpha, foreground, background = Fusion()([imscal, alfgbg])

        model = models.Model(
            inputs=(image, twomap, distance),
            outputs=(alfgbg, alpha, foreground, background),
            name="fba_matting",
        )

        return model

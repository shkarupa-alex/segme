from keras import layers
from segme.common.convnormact import ConvNormAct, Act
from segme.common.ppm import PyramidPooling
from segme.common.sequence import Sequence
from segme.common.resize import BilinearInterpolation
from segme.common.head import HeadProjection


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
        x = Sequence([
            layers.Conv2D(32, 3, padding='same'),
            Act(),
            layers.Conv2D(16, 3, padding='same'),
            Act(),
            HeadProjection(7)
        ])(x)

        return x

    return apply

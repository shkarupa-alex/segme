from keras import layers, models
from segme.model.matting.fba_matting.decoder import Decoder
from segme.model.matting.fba_matting.fusion import Fusion
from segme.model.matting.fba_matting.encoder import Encoder


def FBAMatting():
    image = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    twomap = layers.Input(name='twomap', shape=[None, None, 2], dtype='uint8')
    distance = layers.Input(name='distance', shape=[None, None, 6], dtype='uint8')

    # Rescale twomap and distance to match preprocessed image
    featraw = layers.concatenate([image, twomap, distance], axis=-1)
    feats2, feats4, feats32 = Encoder()(featraw)

    imscal = layers.Rescaling(1 / 255)(image)
    imnorm = layers.Normalization(
        mean=[0.485, 0.456, 0.406], variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2])(imscal)
    alfgbg = Decoder()(
        feats2, feats4, feats32,
        imscal,  # scaled image
        imnorm,  # normalized image
        layers.Rescaling(1 / 255)(twomap)  # scaled twomap
    )

    alfgbg, alpha, foreground, background = Fusion()([imscal, alfgbg])

    model = models.Model(
        inputs=[image, twomap, distance], outputs=[alfgbg, alpha, foreground, background], name='fba_matting')

    return model

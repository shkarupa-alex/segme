import tensorflow as tf
from keras import layers, models
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .encoder import MMFusionEncoder
from .transformer import VisionTransformer
from ...common import HeadActivation, HeadProjection, ResizeByScale, resize_by_scale


@register_keras_serializable(package='SegMe>CascadePSP')
class TriTransNet(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # rgb
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint16')  # depth
        ]

    @shape_type_conversion
    def build(self, input_shape):
        self.encoder = MMFusionEncoder()

        self.transition8 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.transition16 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.transition32 = layers.Conv2D(64, 3, padding='same', activation='relu')

        self.ufm16_up = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.ufm16_proj = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.ufm32_up0 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.ufm32_proj0 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.ufm32_up1 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.ufm32_proj1 = layers.Conv2D(64, 3, padding='same', activation='relu')

        self.vit = VisionTransformer()

        self.deconv8 = models.Sequential([layers.Conv2D(64, 3, padding='same', activation='relu'), ResizeByScale(2)])
        self.deconv16 = models.Sequential([layers.Conv2D(64, 3, padding='same', activation='relu'), ResizeByScale(2)])
        self.deconv32 = models.Sequential([layers.Conv2D(64, 3, padding='same', activation='relu'), ResizeByScale(2)])

        # In original paper first output stride is x4 and it equals to to second one, due to extracted after maxpool.

        # So, the next three sublayers do not have scaling by default ...
        self.deconv48 = models.Sequential([layers.Conv2D(128, 3, padding='same', activation='relu'), ResizeByScale(2)])
        self.deconv416 = models.Sequential([layers.Conv2D(128, 3, padding='same', activation='relu'), ResizeByScale(2)])
        self.deconv432 = models.Sequential([layers.Conv2D(128, 3, padding='same', activation='relu'), ResizeByScale(2)])
        self.deconv248 = models.Sequential([layers.Conv2D(64, 3, padding='same', activation='relu'), ResizeByScale(2)])
        self.deconv2416 = models.Sequential([layers.Conv2D(64, 3, padding='same', activation='relu'), ResizeByScale(2)])
        self.deconv2432 = models.Sequential([layers.Conv2D(64, 3, padding='same', activation='relu'), ResizeByScale(2)])

        self.predict8 = models.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            # ResizeByScale(2),
            HeadProjection(1, kernel_size=3)
        ])
        self.predict16 = models.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            # ResizeByScale(2),
            HeadProjection(1, kernel_size=3)
        ])
        self.predict32 = models.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            # ResizeByScale(2),
            HeadProjection(1, kernel_size=3)
        ])

        self.act = HeadActivation(1)

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        rgb, depth = inputs
        feats2, feats4, feats8, feats16, feats32 = self.encoder([rgb, depth])

        feats8_ts = self.transition8(feats8)
        feats16_ts = self.transition16(feats16)
        feats32_ts = self.transition32(feats32)

        feats16_up = self.ufm16_up(resize_by_scale(feats16_ts, scale=2))
        combo16_8 = layers.concatenate([feats16_up, feats8_ts])
        feats16_proj = self.ufm16_proj(combo16_8)

        feats32_up0 = self.ufm32_up0(resize_by_scale(feats32_ts, scale=2))
        combo32_16 = layers.concatenate([feats32_up0, feats16_ts])
        feats32_proj0 = self.ufm32_proj0(combo32_16)
        feats32_up1 = self.ufm32_up1(resize_by_scale(feats32_proj0, scale=2))
        combo32_8 = layers.concatenate([feats32_up1, feats8_ts])
        feats32_proj1 = self.ufm32_proj1(combo32_8)

        feats8_vit = self.vit(feats8_ts)
        feats16_vit = self.vit(feats16_proj)
        feats32_vit = self.vit(feats32_proj1)

        feats8_vr = layers.concatenate([feats8_vit, feats8_ts])
        feats8_vrd = self.deconv8(feats8_vr)

        feats16_vr = layers.concatenate([feats16_vit, feats16_proj])
        feats16_vrd = self.deconv16(feats16_vr)

        feats32_vr = layers.concatenate([feats32_vit, feats32_proj1])
        feats32_vrd = self.deconv32(feats32_vr)

        feats48_comb = layers.concatenate([feats8_vrd, feats4])
        feats48_combd = self.deconv48(feats48_comb)

        feats248_comb = layers.concatenate([feats48_combd, feats2])
        feats248_combd = self.deconv248(feats248_comb)
        logits8 = self.predict8(feats248_combd)
        preds8 = self.act(logits8)

        feats416_comb = layers.concatenate([feats16_vrd, feats4])
        feats416_comb3 = self.deconv416(feats416_comb)

        feats2416_comb = layers.concatenate([feats416_comb3, feats2])
        feats2416_combd = self.deconv2416(feats2416_comb)
        logits16 = self.predict16(feats2416_combd)
        preds16 = self.act(logits16)

        feats432_comb = layers.concatenate([feats32_vrd, feats4])
        feats432_combd = self.deconv432(feats432_comb)

        feats2432_comb = layers.concatenate([feats432_combd, feats2])
        feats2432_combd = self.deconv2432(feats2432_comb)
        logits32 = self.predict32(feats2432_combd)
        preds32 = self.act(logits32)

        logits = logits8 + logits16 + logits32
        preds = self.act(logits)

        return preds, preds8, preds16, preds32

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return [input_shape[0][:-1] + (1,)] * 4

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return [tf.TensorSpec(dtype='float32', shape=os.shape) for os in outptut_signature]


def build_tri_trans_net(image_size):
    inputs = [
        layers.Input(name='image', shape=[image_size, image_size, 3], dtype='uint8'),
        layers.Input(name='depth', shape=[image_size, image_size, 1], dtype='uint16')
    ]
    outputs = TriTransNet()(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='tri_trans_net')

    return model

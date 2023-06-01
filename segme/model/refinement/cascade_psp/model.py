import tensorflow as tf
from keras import backend, layers, models
from keras.saving import register_keras_serializable
from keras.src.utils.control_flow_util import smart_cond
from keras.src.utils.tf_utils import shape_type_conversion
from segme.common.convnormact import Conv, Act
from segme.common.ppm import PyramidPooling
from segme.common.sequence import Sequence
from segme.common.head import HeadProjection, ClassificationActivation
from segme.common.resize import BilinearInterpolation
from segme.model.refinement.cascade_psp.upsample import Upsample
from segme.model.refinement.cascade_psp.encoder import Encoder


@register_keras_serializable(package='SegMe>Model>Refinement>CascadePSP')
class CascadePSP(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # image
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8'),  # mask
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8'),  # previous prediction
        ]

    @shape_type_conversion
    def build(self, input_shape):
        self.bone = Encoder()
        self.intbysamp = BilinearInterpolation(None)

        self.psp = PyramidPooling(1024)

        self.up1 = Upsample(512)
        self.up2 = Upsample(256)
        self.up3 = Upsample(32)

        self.final8 = Sequence([
            Conv(32, 1),
            Act(),
            HeadProjection(1)
        ])

        self.final4 = Sequence([
            Conv(32, 1),
            Act(),
            HeadProjection(1)
        ])
        self.final1 = Sequence([
            Conv(32, 1),
            Act(),
            HeadProjection(1)
        ])

        self.act = ClassificationActivation()

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = backend.learning_phase()

        image, mask, prev = inputs
        no_prev = tf.logical_or(tf.cast(training, 'bool'), tf.reduce_all(tf.equal(mask, prev)))

        image = tf.cast(image, self.compute_dtype)
        mask = tf.cast(mask, self.compute_dtype)
        prev = tf.cast(prev, self.compute_dtype)

        """
        First iteration, s8 output
        """
        pred_s8_1, scale_s8_1 = smart_cond(
            no_prev,
            lambda: self._estimate_s8(image, mask),
            lambda: (tf.cast(prev / 255., 'float32'), tf.identity(prev)))

        """
        Second iteration, s4 output
        """
        inp2 = tf.concat([image, mask, scale_s8_1, scale_s8_1], axis=-1)
        _, feats4, feats8 = self.bone(inp2)

        out2_s8 = self.psp(feats8)
        mask_s8_2 = self.final8(out2_s8)
        mask_s8_2 = self.intbysamp([mask_s8_2, image])
        pred_s8_2 = self.act(mask_s8_2)
        scale_s8_2 = tf.cast(pred_s8_2 * 255., self.compute_dtype)

        out2_s4 = self.up1([out2_s8, feats4])
        mask_s4_2 = self.final4(out2_s4)
        mask_s4_2 = self.intbysamp([mask_s4_2, image])
        pred_s4_2 = self.act(mask_s4_2)
        scale_s4_2 = tf.cast(pred_s4_2 * 255., self.compute_dtype)

        """
        Third iteration, s1 output
        """
        inp3 = tf.concat([image, mask, scale_s8_2, scale_s4_2], axis=-1)

        feats2, feats4, feats8 = self.bone(inp3)
        out3_s8 = self.psp(feats8)

        mask_s8_3 = self.final8(out3_s8)
        mask_s8_3 = self.intbysamp([mask_s8_3, image])
        pred_s8_3 = self.act(mask_s8_3)

        out3_s4 = self.up1([out3_s8, feats4])
        mask_s4_3 = self.final4(out3_s4)
        mask_s4_3 = self.intbysamp([mask_s4_3, image])
        pred_s4_3 = self.act(mask_s4_3)

        out3_s4 = self.up2([out3_s4, feats2])
        out3_s4 = self.up3([out3_s4, image])

        mask_s1_3 = self.final1(tf.concat([out3_s4, image], axis=-1))
        pred_s1_3 = self.act(mask_s1_3)

        """
        Final output
        """
        preds = [pred_s1_3, pred_s4_3, pred_s8_3, pred_s4_2, pred_s8_2, pred_s8_1]

        return preds

    def _estimate_s8(self, image, mask):
        inp1 = tf.concat([image, mask, mask, mask], axis=-1)
        _, _, feats8 = self.bone(inp1)

        out1 = self.psp(feats8)
        mask_s8 = self.final8(out1)
        mask_s8 = self.intbysamp([mask_s8, image])
        pred_s8 = self.act(mask_s8)
        scale_s8 = tf.cast(pred_s8 * 255., self.compute_dtype)

        return pred_s8, scale_s8

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return [input_shape[0][:-1] + (1,)] * 6

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return [tf.TensorSpec(dtype='float32', shape=os.shape) for os in outptut_signature]


def build_cascade_psp():
    inputs = [
        layers.Input(name='image', shape=[None, None, 3], dtype='uint8'),
        layers.Input(name='mask', shape=[None, None, 1], dtype='uint8'),
        layers.Input(name='prev', shape=[None, None, 1], dtype='uint8'),
    ]
    outputs = CascadePSP()(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='cascade_psp')

    return model

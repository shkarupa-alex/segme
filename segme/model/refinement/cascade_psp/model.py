import tensorflow as tf
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable

from segme.common.convnormact import Act
from segme.common.convnormact import Conv
from segme.common.head import ClassificationActivation
from segme.common.head import HeadProjection
from segme.common.ppm import PyramidPooling
from segme.common.resize import BilinearInterpolation
from segme.common.sequence import Sequence
from segme.model.refinement.cascade_psp.encoder import Encoder
from segme.model.refinement.cascade_psp.upsample import Upsample


@register_keras_serializable(package="SegMe>Model>Refinement>CascadePSP")
class CascadePSP(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            InputSpec(ndim=4, axes={-1: 3}, dtype="uint8"),  # image
            InputSpec(ndim=4, axes={-1: 1}, dtype="uint8"),  # mask
            InputSpec(
                ndim=4, axes={-1: 1}, dtype="uint8"
            ),  # previous prediction
        ]

    def build(self, input_shape):
        self.bone = Encoder(dtype=self.dtype_policy)
        self.intbysamp = BilinearInterpolation(dtype=self.dtype_policy)

        self.psp = PyramidPooling(1024, dtype=self.dtype_policy)

        self.up1 = Upsample(512, dtype=self.dtype_policy)
        self.up2 = Upsample(256, dtype=self.dtype_policy)
        self.up3 = Upsample(32, dtype=self.dtype_policy)

        self.final8 = Sequence(
            [
                Conv(32, 1, dtype=self.dtype_policy),
                Act(dtype=self.dtype_policy),
                HeadProjection(1, dtype=self.dtype_policy),
            ],
            dtype=self.dtype_policy,
        )

        self.final4 = Sequence(
            [
                Conv(32, 1, dtype=self.dtype_policy),
                Act(dtype=self.dtype_policy),
                HeadProjection(1, dtype=self.dtype_policy),
            ],
            dtype=self.dtype_policy,
        )
        self.final1 = Sequence(
            [
                Conv(32, 1, dtype=self.dtype_policy),
                Act(dtype=self.dtype_policy),
                HeadProjection(1, dtype=self.dtype_policy),
            ],
            dtype=self.dtype_policy,
        )

        self.act = ClassificationActivation()

        super().build(input_shape)

    def call(self, inputs, training=False, **kwargs):
        image, mask, prev = inputs
        no_prev = tf.logical_or(
            ops.cast(training, "bool"), ops.all(tf.equal(mask, prev))
        )

        image = ops.cast(image, self.compute_dtype)
        mask = ops.cast(mask, self.compute_dtype)
        prev = ops.cast(prev, self.compute_dtype)

        """
        First iteration, s8 output
        """
        pred_s8_1, scale_s8_1 = ops.cond(
            no_prev,
            lambda: self._estimate_s8(image, mask),
            lambda: (ops.cast(prev / 255.0, "float32"), prev),
        )

        """
        Second iteration, s4 output
        """
        inp2 = ops.concatenate([image, mask, scale_s8_1, scale_s8_1], axis=-1)
        _, feats4, feats8 = self.bone(inp2)

        out2_s8 = self.psp(feats8)
        mask_s8_2 = self.final8(out2_s8)
        mask_s8_2 = self.intbysamp([mask_s8_2, image])
        pred_s8_2 = self.act(mask_s8_2)
        scale_s8_2 = ops.cast(pred_s8_2 * 255.0, self.compute_dtype)

        out2_s4 = self.up1([out2_s8, feats4])
        mask_s4_2 = self.final4(out2_s4)
        mask_s4_2 = self.intbysamp([mask_s4_2, image])
        pred_s4_2 = self.act(mask_s4_2)
        scale_s4_2 = ops.cast(pred_s4_2 * 255.0, self.compute_dtype)

        """
        Third iteration, s1 output
        """
        inp3 = ops.concatenate([image, mask, scale_s8_2, scale_s4_2], axis=-1)

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

        mask_s1_3 = self.final1(ops.concatenate([out3_s4, image], axis=-1))
        pred_s1_3 = self.act(mask_s1_3)

        """
        Final output
        """
        preds = [
            pred_s1_3,
            pred_s4_3,
            pred_s8_3,
            pred_s4_2,
            pred_s8_2,
            pred_s8_1,
        ]

        return preds

    def _estimate_s8(self, image, mask):
        inp1 = ops.concatenate([image, mask, mask, mask], axis=-1)
        _, _, feats8 = self.bone(inp1)

        out1 = self.psp(feats8)
        mask_s8 = self.final8(out1)
        mask_s8 = self.intbysamp([mask_s8, image])
        pred_s8 = self.act(mask_s8)
        scale_s8 = ops.cast(pred_s8 * 255.0, self.compute_dtype)

        return pred_s8, scale_s8

    def compute_output_shape(self, input_shape):
        return [input_shape[0][:-1] + (1,)] * 6

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return [
            tf.TensorSpec(dtype="float32", shape=os.shape)
            for os in outptut_signature
        ]


def build_cascade_psp():
    inputs = [
        layers.Input(name="image", shape=(None, None, 3), dtype="uint8"),
        layers.Input(name="mask", shape=(None, None, 1), dtype="uint8"),
        layers.Input(name="prev", shape=(None, None, 1), dtype="uint8"),
    ]
    outputs = CascadePSP()(inputs)
    model = models.Functional(
        inputs=inputs, outputs=outputs, name="cascade_psp"
    )

    return model

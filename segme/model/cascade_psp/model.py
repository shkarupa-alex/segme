import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers, utils
from tensorflow.python.keras.utils.control_flow_util import smart_cond
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .psp import PSP
from .upsample import Upsample
from .resnet import ResNet50
from ...common import HeadActivation, HeadProjection, ResizeBySample


@utils.register_keras_serializable(package='SegMe>CascadePSP')
class CascadePSP(layers.Layer):
    def __init__(self, psp_sizes, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(ndim=4, axes={-1: 3}, dtype='uint8'),  # image
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8'),  # mask
            layers.InputSpec(ndim=4, axes={-1: 1}, dtype='uint8'),  # previous prediction
        ]
        self.psp_sizes = psp_sizes

    @shape_type_conversion
    def build(self, input_shape):
        self.bone = ResNet50()

        self.resize = ResizeBySample(align_corners=False)

        self.psp = PSP(1024, self.psp_sizes)

        self.up1 = Upsample(512)
        self.up2 = Upsample(256)
        self.up3 = Upsample(32)

        self.final8 = Sequential([
            layers.Conv2D(32, 1, padding='same'),
            layers.ReLU(),
            HeadProjection(1)
        ])
        self.final4 = Sequential([
            layers.Conv2D(32, 1, padding='same'),
            layers.ReLU(),
            HeadProjection(1)
        ])
        self.final1 = Sequential([
            layers.Conv2D(32, 1, padding='same'),
            layers.ReLU(),
            HeadProjection(1)
        ])

        self.act = HeadActivation(1)

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            training = tf.keras.backend.learning_phase()

        image, mask, prev = inputs
        image = self._preprocess_image(image)
        mask = self._preprocess_mask(mask)
        prev = self._preprocess_mask(prev)

        """
        First iteration, s8 output
        """
        is_prev = (not training) & (tf.reduce_max(prev - mask) > tf.keras.backend.epsilon())
        mask_s8_1 = smart_cond(is_prev, lambda: prev, lambda: self._estimate_s8(image, mask))

        """
        Second iteration, s4 output
        """
        inp2 = layers.concatenate([image, mask, mask_s8_1, mask_s8_1])
        _, feats4, feats8 = self.bone(inp2)

        out2_s8 = self.psp(feats8)
        mask_s8_2 = self.final8(out2_s8)
        mask_s8_2 = self.resize([mask_s8_2, image])

        out2_s4 = self.up1([out2_s8, feats4])
        mask_s4_2 = self.final4(out2_s4)
        mask_s4_2 = self.resize([mask_s4_2, image])

        """
        Third iteration, s1 output
        """
        inp3 = layers.concatenate([image, mask, mask_s8_2, mask_s4_2])

        feats2, feats4, feats8 = self.bone(inp3)
        out3_s8 = self.psp(feats8)

        mask_s8_3 = self.final8(out3_s8)
        mask_s8_3 = self.resize([mask_s8_3, image])

        out3_s4 = self.up1([out3_s8, feats4])
        mask_s4_3 = self.final4(out3_s4)
        mask_s4_3 = self.resize([mask_s4_3, image])

        out3_s4 = self.up2([out3_s4, feats2])
        out3_s4 = self.up3([out3_s4, image])

        mask_s1_3 = self.final1(layers.concatenate([out3_s4, image]))

        """
        Final output
        """
        # Originally: pred_224 pred_56_2 pred_28_3 pred_56 pred_28_2 pred_28
        preds = [mask_s1_3, mask_s4_3, mask_s8_3, mask_s4_2, mask_s8_2, mask_s8_1]
        preds = [self.act(p) for p in preds]

        return preds

    def _preprocess_image(self, image):
        image = tf.cast(image, self.compute_dtype)
        image = image[..., ::-1]  # 'RGB'->'BGR'
        image = tf.nn.bias_add(image, [-103.939, -116.779, -123.68])

        return image

    def _preprocess_mask(self, mask):
        mask = tf.cast(mask, self.compute_dtype)
        mask = tf.nn.bias_add(mask, [-127.])

        return mask

    def _estimate_s8(self, image, mask):
        inp1 = layers.concatenate([image, mask, mask, mask])
        _, _, feats8 = self.bone(inp1)

        out1 = self.psp(feats8)
        mask_s8 = self.final8(out1)
        mask_s8 = self.resize([mask_s8, image])

        return mask_s8

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return [input_shape[0][:-1] + (1,)] * 6

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return [tf.TensorSpec(dtype='float32', shape=os.shape) for os in outptut_signature]

    def get_config(self):
        config = super().get_config()
        config.update({'psp_sizes': self.psp_sizes})

        return config


def build_cascade_psp(psp_sizes=(1, 2, 3, 6)):
    inputs = [
        layers.Input(name='image', shape=[None, None, 3], dtype='uint8'),
        layers.Input(name='mask', shape=[None, None, 1], dtype='uint8'),
        layers.Input(name='prev', shape=[None, None, 1], dtype='uint8'),
    ]
    outputs = CascadePSP(psp_sizes)(inputs)
    model = Model(inputs=inputs, outputs=outputs, name='cascade_psp')

    return model

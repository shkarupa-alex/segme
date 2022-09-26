import tensorflow as tf
from keras import layers, models
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from segme.common.head import HeadProjection, ClassificationActivation, ClassificationHead
from segme.common.pad import SymmetricPadding
from segme.common.interrough import BilinearInterpolation
from segme.common.sequent import Sequential
from segme.model.sod.u2_net.rsu7 import RSU7
from segme.model.sod.u2_net.rsu6 import RSU6
from segme.model.sod.u2_net.rsu5 import RSU5
from segme.model.sod.u2_net.rsu4 import RSU4
from segme.model.sod.u2_net.rsu4f import RSU4F


@register_keras_serializable(package='SegMe>Model>SOD>U2Net')
class U2Net(layers.Layer):
    """ Reference: https://arxiv.org/pdf/2005.09007.pdf """

    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')

        self.classes = classes

    @shape_type_conversion
    def build(self, input_shape):
        self.pool = layers.MaxPool2D(2, padding='same')
        self.resize = BilinearInterpolation(None)

        self.stage1 = RSU7(32, 64)
        self.stage2 = RSU6(32, 128)
        self.stage3 = RSU5(64, 256)
        self.stage4 = RSU4(128, 512)
        self.stage5 = RSU4F(256, 512)
        self.stage6 = RSU4F(256, 512)

        self.stage5d = RSU4F(256, 512)
        self.stage4d = RSU4(128, 256)
        self.stage3d = RSU5(64, 128)
        self.stage2d = RSU6(32, 64)
        self.stage1d = RSU7(16, 64)

        self.proj1 = HeadProjection(self.classes, 3)
        self.proj2 = HeadProjection(self.classes, 3)
        self.proj3 = HeadProjection(self.classes, 3)
        self.proj4 = HeadProjection(self.classes, 3)
        self.proj5 = HeadProjection(self.classes, 3)
        self.proj6 = HeadProjection(self.classes, 3)
        self.act = ClassificationActivation()

        self.head = ClassificationHead(self.classes)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = preprocess_input(tf.cast(inputs, self.compute_dtype), mode='torch')

        outputs1 = self.stage1(outputs)
        outputs = self.pool(outputs1)

        outputs2 = self.stage2(outputs)
        outputs = self.pool(outputs2)

        outputs3 = self.stage3(outputs)
        outputs = self.pool(outputs3)

        outputs4 = self.stage4(outputs)
        outputs = self.pool(outputs4)

        outputs5 = self.stage5(outputs)
        outputs = self.pool(outputs5)

        outputs6 = self.stage6(outputs)
        hx6up = self.resize([outputs6, outputs5])

        # decoder
        outputs5d = self.stage5d(tf.concat([hx6up, outputs5], axis=-1))
        outputs5dup = self.resize([outputs5d, outputs4])

        outputs4d = self.stage4d(tf.concat([outputs5dup, outputs4], axis=-1))
        outputs4dup = self.resize([outputs4d, outputs3])

        outputs3d = self.stage3d(tf.concat([outputs4dup, outputs3], axis=-1))
        outputs3dup = self.resize([outputs3d, outputs2])

        outputs2d = self.stage2d(tf.concat([outputs3dup, outputs2], axis=-1))
        outputs2dup = self.resize([outputs2d, outputs1])

        outputs1d = self.stage1d(tf.concat([outputs2dup, outputs1], axis=-1))

        # side output
        n1 = self.proj1(outputs1d)

        n2 = self.proj2(outputs2d)
        n2 = self.resize([n2, n1])

        n3 = self.proj3(outputs3d)
        n3 = self.resize([n3, n1])

        n4 = self.proj4(outputs4d)
        n4 = self.resize([n4, n1])

        n5 = self.proj5(outputs5d)
        n5 = self.resize([n5, n1])

        n6 = self.proj6(outputs6)
        n6 = self.resize([n6, n1])

        h = self.head(tf.concat([n1, n2, n3, n4, n5, n6], axis=-1))
        h1 = self.act(n1)
        h2 = self.act(n2)
        h3 = self.act(n3)
        h4 = self.act(n4)
        h5 = self.act(n5)
        h6 = self.act(n6)

        return h, h1, h2, h3, h4, h5, h6

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return (self.head.compute_output_shape(input_shape),) * 7

    def compute_output_signature(self, input_signature):
        return (self.head.compute_output_signature(input_signature),) * 7

    def get_config(self):
        config = super().get_config()
        config.update({'classes': self.classes})

        return config


@register_keras_serializable(package='SegMe>Model>SOD>U2Net')
class U2NetP(layers.Layer):
    """ Reference: https://arxiv.org/pdf/2005.09007.pdf """

    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')
        self.classes = classes

    @shape_type_conversion
    def build(self, input_shape):
        self.pool = layers.MaxPool2D(2, padding='same')
        self.resize = BilinearInterpolation(None)

        self.stage1 = RSU7(16, 64)
        self.stage2 = RSU6(16, 64)
        self.stage3 = RSU5(16, 64)
        self.stage4 = RSU4(16, 64)
        self.stage5 = RSU4F(16, 64)
        self.stage6 = RSU4F(16, 64)

        self.stage5d = RSU4F(16, 64)
        self.stage4d = RSU4(16, 64)
        self.stage3d = RSU5(16, 64)
        self.stage2d = RSU6(16, 64)
        self.stage1d = RSU7(16, 64)

        self.proj1 = HeadProjection(self.classes, 3)
        self.proj2 = HeadProjection(self.classes, 3)
        self.proj3 = HeadProjection(self.classes, 3)
        self.proj4 = HeadProjection(self.classes, 3)
        self.proj5 = HeadProjection(self.classes, 3)
        self.proj6 = HeadProjection(self.classes, 3)
        self.act = ClassificationActivation()

        self.head = ClassificationHead(self.classes)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = preprocess_input(tf.cast(inputs, self.compute_dtype), mode='torch')

        outputs1 = self.stage1(outputs)
        outputs = self.pool(outputs1)

        outputs2 = self.stage2(outputs)
        outputs = self.pool(outputs2)

        outputs3 = self.stage3(outputs)
        outputs = self.pool(outputs3)

        outputs4 = self.stage4(outputs)
        outputs = self.pool(outputs4)

        outputs5 = self.stage5(outputs)
        outputs = self.pool(outputs5)

        outputs6 = self.stage6(outputs)
        hx6up = self.resize([outputs6, outputs5])

        # decoder
        outputs5d = self.stage5d(tf.concat([hx6up, outputs5], axis=-1))
        outputs5dup = self.resize([outputs5d, outputs4])

        outputs4d = self.stage4d(tf.concat([outputs5dup, outputs4], axis=-1))
        outputs4dup = self.resize([outputs4d, outputs3])

        outputs3d = self.stage3d(tf.concat([outputs4dup, outputs3], axis=-1))
        outputs3dup = self.resize([outputs3d, outputs2])

        outputs2d = self.stage2d(tf.concat([outputs3dup, outputs2], axis=-1))
        outputs2dup = self.resize([outputs2d, outputs1])

        outputs1d = self.stage1d(tf.concat([outputs2dup, outputs1], axis=-1))

        # side output
        n1 = self.proj1(outputs1d)

        n2 = self.proj2(outputs2d)
        n2 = self.resize([n2, n1])

        n3 = self.proj3(outputs3d)
        n3 = self.resize([n3, n1])

        n4 = self.proj4(outputs4d)
        n4 = self.resize([n4, n1])

        n5 = self.proj5(outputs5d)
        n5 = self.resize([n5, n1])

        n6 = self.proj6(outputs6)
        n6 = self.resize([n6, n1])

        h = self.head(tf.concat([n1, n2, n3, n4, n5, n6], axis=-1))
        h1 = self.act(n1)
        h2 = self.act(n2)
        h3 = self.act(n3)
        h4 = self.act(n4)
        h5 = self.act(n5)
        h6 = self.act(n6)

        return h, h1, h2, h3, h4, h5, h6

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return (self.head.compute_output_shape(input_shape),) * 7

    def compute_output_signature(self, input_signature):
        return (self.head.compute_output_signature(input_signature),) * 7

    def get_config(self):
        config = super().get_config()
        config.update({'classes': self.classes})

        return config


def build_u2_net(classes):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = U2Net(classes)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='u2_net')

    return model


def build_u2_netp(classes):
    inputs = layers.Input(name='image', shape=[None, None, 3], dtype='uint8')
    outputs = U2NetP(classes)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name='u2_netp')

    return model

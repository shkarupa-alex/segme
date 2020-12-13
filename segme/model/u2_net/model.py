import tensorflow as tf
from tensorflow.keras import Model, layers, utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .rsu7 import RSU7
from .rsu6 import RSU6
from .rsu5 import RSU5
from .rsu4 import RSU4
from .rsu4f import RSU4F
from ...common import HeadProjection, HeadActivation, ClassificationHead, resize_by_sample


@utils.register_keras_serializable(package='SegMe>U2Net')
class U2Net(layers.Layer):
    """ Reference: https://arxiv.org/pdf/2005.09007.pdf """

    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')
        self.classes = classes

    @shape_type_conversion
    def build(self, input_shape):
        self.prep = layers.Lambda(
            lambda img: preprocess_input(tf.cast(img, tf.float32), 'channels_last', 'tf'), name='preprocess')

        self.stage1 = RSU7(32, 64)
        self.pool12 = layers.MaxPool2D(2, padding='same')

        self.stage2 = RSU6(32, 128)
        self.pool23 = layers.MaxPool2D(2, padding='same')

        self.stage3 = RSU5(64, 256)
        self.pool34 = layers.MaxPool2D(2, padding='same')

        self.stage4 = RSU4(128, 512)
        self.pool45 = layers.MaxPool2D(2, padding='same')

        self.stage5 = RSU4F(256, 512)
        self.pool56 = layers.MaxPool2D(2, padding='same')

        self.stage6 = RSU4F(256, 512)

        # decoder
        self.stage5d = RSU4F(256, 512)
        self.stage4d = RSU4(128, 256)
        self.stage3d = RSU5(64, 128)
        self.stage2d = RSU6(32, 64)
        self.stage1d = RSU7(16, 64)

        self.proj1 = HeadProjection(self.classes, kernel_size=3)
        self.proj2 = HeadProjection(self.classes, kernel_size=3)
        self.proj3 = HeadProjection(self.classes, kernel_size=3)
        self.proj4 = HeadProjection(self.classes, kernel_size=3)
        self.proj5 = HeadProjection(self.classes, kernel_size=3)
        self.proj6 = HeadProjection(self.classes, kernel_size=3)
        self.act = HeadActivation(self.classes)
        self.head = ClassificationHead(self.classes)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.prep(inputs)

        # stage 1
        outputs1 = self.stage1(outputs)
        outputs = self.pool12(outputs1)

        # stage 2
        outputs2 = self.stage2(outputs)
        outputs = self.pool23(outputs2)

        # stage 3
        outputs3 = self.stage3(outputs)
        outputs = self.pool34(outputs3)

        # stage 4
        outputs4 = self.stage4(outputs)
        outputs = self.pool45(outputs4)

        # stage 5
        outputs5 = self.stage5(outputs)
        outputs = self.pool56(outputs5)

        # stage 6
        outputs6 = self.stage6(outputs)
        hx6up = resize_by_sample([outputs6, outputs5])

        # -------------------- decoder --------------------
        outputs5d = self.stage5d(layers.concatenate([hx6up, outputs5]))
        outputs5dup = resize_by_sample([outputs5d, outputs4])

        outputs4d = self.stage4d(layers.concatenate([outputs5dup, outputs4]))
        outputs4dup = resize_by_sample([outputs4d, outputs3])

        outputs3d = self.stage3d(layers.concatenate([outputs4dup, outputs3]))
        outputs3dup = resize_by_sample([outputs3d, outputs2])

        outputs2d = self.stage2d(layers.concatenate([outputs3dup, outputs2]))
        outputs2dup = resize_by_sample([outputs2d, outputs1])

        outputs1d = self.stage1d(layers.concatenate([outputs2dup, outputs1]))

        # side output
        n1 = self.proj1(outputs1d)

        n2 = self.proj2(outputs2d)
        n2 = resize_by_sample([n2, n1])

        n3 = self.proj3(outputs3d)
        n3 = resize_by_sample([n3, n1])

        n4 = self.proj4(outputs4d)
        n4 = resize_by_sample([n4, n1])

        n5 = self.proj5(outputs5d)
        n5 = resize_by_sample([n5, n1])

        n6 = self.proj6(outputs6)
        n6 = resize_by_sample([n6, n1])

        h = self.head(layers.concatenate([n1, n2, n3, n4, n5, n6]))
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


@utils.register_keras_serializable(package='SegMe>U2Net')
class U2NetP(layers.Layer):
    """ Reference: https://arxiv.org/pdf/2005.09007.pdf """

    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')
        self.classes = classes

    @shape_type_conversion
    def build(self, input_shape):
        self.prep = layers.Lambda(
            lambda img: preprocess_input(tf.cast(img, tf.float32), 'channels_last', 'caffe'), name='preprocess')

        self.stage1 = RSU7(16, 64)
        self.pool12 = layers.MaxPool2D(2, padding='same')

        self.stage2 = RSU6(16, 64)
        self.pool23 = layers.MaxPool2D(2, padding='same')

        self.stage3 = RSU5(16, 64)
        self.pool34 = layers.MaxPool2D(2, padding='same')

        self.stage4 = RSU4(16, 64)
        self.pool45 = layers.MaxPool2D(2, padding='same')

        self.stage5 = RSU4F(16, 64)
        self.pool56 = layers.MaxPool2D(2, padding='same')

        self.stage6 = RSU4F(16, 64)

        # decoder
        self.stage5d = RSU4F(16, 64)
        self.stage4d = RSU4(16, 64)
        self.stage3d = RSU5(16, 64)
        self.stage2d = RSU6(16, 64)
        self.stage1d = RSU7(16, 64)

        self.proj1 = HeadProjection(self.classes, kernel_size=3)
        self.proj2 = HeadProjection(self.classes, kernel_size=3)
        self.proj3 = HeadProjection(self.classes, kernel_size=3)
        self.proj4 = HeadProjection(self.classes, kernel_size=3)
        self.proj5 = HeadProjection(self.classes, kernel_size=3)
        self.proj6 = HeadProjection(self.classes, kernel_size=3)
        self.act = HeadActivation(self.classes)
        self.head = ClassificationHead(self.classes)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.prep(inputs)

        # stage 1
        outputs1 = self.stage1(outputs)
        outputs = self.pool12(outputs1)

        # stage 2
        outputs2 = self.stage2(outputs)
        outputs = self.pool23(outputs2)

        # stage 3
        outputs3 = self.stage3(outputs)
        outputs = self.pool34(outputs3)

        # stage 4
        outputs4 = self.stage4(outputs)
        outputs = self.pool45(outputs4)

        # stage 5
        outputs5 = self.stage5(outputs)
        outputs = self.pool56(outputs5)

        # stage 6
        outputs6 = self.stage6(outputs)
        hx6up = resize_by_sample([outputs6, outputs5])

        # decoder
        outputs5d = self.stage5d(layers.concatenate([hx6up, outputs5]))
        outputs5dup = resize_by_sample([outputs5d, outputs4])

        outputs4d = self.stage4d(layers.concatenate([outputs5dup, outputs4]))
        outputs4dup = resize_by_sample([outputs4d, outputs3])

        outputs3d = self.stage3d(layers.concatenate([outputs4dup, outputs3]))
        outputs3dup = resize_by_sample([outputs3d, outputs2])

        outputs2d = self.stage2d(layers.concatenate([outputs3dup, outputs2]))
        outputs2dup = resize_by_sample([outputs2d, outputs1])

        outputs1d = self.stage1d(layers.concatenate([outputs2dup, outputs1]))

        # side output
        n1 = self.proj1(outputs1d)

        n2 = self.proj2(outputs2d)
        n2 = resize_by_sample([n2, n1])

        n3 = self.proj3(outputs3d)
        n3 = resize_by_sample([n3, n1])

        n4 = self.proj4(outputs4d)
        n4 = resize_by_sample([n4, n1])

        n5 = self.proj5(outputs5d)
        n5 = resize_by_sample([n5, n1])

        n6 = self.proj6(outputs6)
        n6 = resize_by_sample([n6, n1])

        h = self.head(layers.concatenate([n1, n2, n3, n4, n5, n6]))
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


def build_u2_net(channels, classes):
    inputs = layers.Input(name='image', shape=[None, None, channels], dtype='uint8')
    outputs = U2Net(classes)(inputs)
    model = Model(inputs=inputs, outputs=outputs, name='u2_net')

    return model


def build_u2_netp(channels, classes):
    inputs = layers.Input(name='image', shape=[None, None, channels], dtype='uint8')
    outputs = U2NetP(classes)(inputs)
    model = Model(inputs=inputs, outputs=outputs, name='u2_netp')

    return model

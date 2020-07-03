import tensorflow as tf
from tensorflow.keras import layers, utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .dense import DenseBlock
from .upconv import UpConvBlock
from .single import SingleConvBlock
from .double import DoubleConvBlock
from ...common import ClassificationHead


@utils.register_keras_serializable(package='SegMe')
class DexiNed(layers.Layer):
    """ Reference: https://arxiv.org/pdf/1909.01955.pdf """

    def __init__(self, classes=2, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')
        self.classes = classes

    @shape_type_conversion
    def build(self, input_shape):
        self.prep = layers.Lambda(
            lambda img: preprocess_input(tf.cast(img, tf.float32), 'channels_last', 'tf'), name='preprocess')

        self.block_1 = DoubleConvBlock(32, 64, stride=2)
        self.block_2 = DoubleConvBlock(128)
        self.dblock_3 = DenseBlock(2, 256)
        self.dblock_4 = DenseBlock(3, 512)
        self.dblock_5 = DenseBlock(3, 512)
        self.dblock_6 = DenseBlock(3, 256)
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        # first skip connection
        self.side_1 = SingleConvBlock(128, stride=2)
        self.side_2 = SingleConvBlock(256, stride=2)
        self.side_3 = SingleConvBlock(512, stride=2)
        self.side_4 = SingleConvBlock(512)

        self.pre_dense_2 = SingleConvBlock(256, stride=2, weight_norm=False)
        self.pre_dense_3 = SingleConvBlock(256)
        self.pre_dense_4 = SingleConvBlock(512)
        self.pre_dense_5_0 = SingleConvBlock(512, stride=2, weight_norm=False)
        self.pre_dense_5 = SingleConvBlock(512)
        self.pre_dense_6 = SingleConvBlock(256)

        self.up_block_1 = UpConvBlock(1)
        self.up_block_2 = UpConvBlock(1)
        self.up_block_3 = UpConvBlock(2)
        self.up_block_4 = UpConvBlock(3)
        self.up_block_5 = UpConvBlock(4)
        self.up_block_6 = UpConvBlock(4)

        if self.classes:
            self.head = ClassificationHead(self.classes, kernel_size=3, kernel_initializer=tf.constant_initializer(0.2))

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.prep(inputs)

        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = layers.add([block_2_down, block_1_side])
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_down = self.maxpool(block_3)
        block_3_add = layers.add([block_3_down, block_2_side])
        block_3_side = self.side_3(block_3_add)

        # Block 4
        block_4_pre_dense_256 = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(layers.add([block_4_pre_dense_256, block_3_down]))
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        # Block 5
        block_5_pre_dense_512 = self.pre_dense_5_0(block_4_pre_dense_256)
        block_5_pre_dense = self.pre_dense_5(layers.add([block_5_pre_dense_512, block_4_down]))
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = layers.add([block_5, block_4_side])

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        # upsampling blocks
        scales = [
            self.up_block_1(block_1),
            self.up_block_2(block_2),
            self.up_block_3(block_3),
            self.up_block_4(block_4),
            self.up_block_5(block_5),
            self.up_block_6(block_6),
        ]

        # concatenate multiscale outputs
        outputs = layers.concatenate(scales)  # BxHxWX6
        if not self.classes:
            return outputs

        outputs = self.head(outputs)  # BxHxWX1

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if not self.classes:
            return input_shape[:-1] + (6,)

        return self.head.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({'classes': self.classes})

        return config

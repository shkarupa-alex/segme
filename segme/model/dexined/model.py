import tensorflow as tf
from tensorflow.keras import layers, utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion
from .dense import DexiNedDenseBlock
from .upconv import DexiNedUpConvBlock
from .single import DexiNedSingleConvBlock
from .double import DexiNedDoubleConvBlock


@utils.register_keras_serializable(package='SegMe')
class DexiNed(layers.Layer):
    """ Reference: https://arxiv.org/pdf/1909.01955.pdf """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')

    @shape_type_conversion
    def build(self, input_shape):
        self.prep = layers.Lambda(
            lambda img: preprocess_input(
                tf.cast(img, tf.float32), 'channels_last', 'tf'),
            name='preprocess')

        self.block_1 = DexiNedDoubleConvBlock(32, 64, stride=2)
        self.block_2 = DexiNedDoubleConvBlock(128)
        self.dblock_3 = DexiNedDenseBlock(2, 256)
        self.dblock_4 = DexiNedDenseBlock(3, 512)
        self.dblock_5 = DexiNedDenseBlock(3, 512)
        self.dblock_6 = DexiNedDenseBlock(3, 256)
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        # first skip connection
        self.side_1 = DexiNedSingleConvBlock(
            128, kernel_size=1, stride=2, weight_norm=True)
        self.side_2 = DexiNedSingleConvBlock(
            256, kernel_size=1, stride=2, weight_norm=True)
        self.side_3 = DexiNedSingleConvBlock(
            512, kernel_size=1, stride=2, weight_norm=True)
        self.side_4 = DexiNedSingleConvBlock(
            512, kernel_size=1, stride=1, weight_norm=True)

        self.pre_dense_2 = DexiNedSingleConvBlock(
            256, kernel_size=1, stride=2)  # use_bn=True
        self.pre_dense_3 = DexiNedSingleConvBlock(
            256, kernel_size=1, stride=1, weight_norm=True)
        self.pre_dense_4 = DexiNedSingleConvBlock(
            512, kernel_size=1, stride=1, weight_norm=True)
        self.pre_dense_5_0 = DexiNedSingleConvBlock(
            512, kernel_size=1, stride=2)  # use_bn=True
        self.pre_dense_5 = DexiNedSingleConvBlock(
            512, kernel_size=1, stride=1, weight_norm=True)
        self.pre_dense_6 = DexiNedSingleConvBlock(
            256, kernel_size=1, stride=1, weight_norm=True)

        # bias_initializer = tf.constant_initializer(-1.996)
        self.up_block_1 = DexiNedUpConvBlock(1)
        self.up_block_2 = DexiNedUpConvBlock(1)
        self.up_block_3 = DexiNedUpConvBlock(2)
        self.up_block_4 = DexiNedUpConvBlock(3)
        self.up_block_5 = DexiNedUpConvBlock(4)
        self.up_block_6 = DexiNedUpConvBlock(4)

        # bias_initializer = tf.constant_initializer(-1.996)
        self.block_cat = DexiNedSingleConvBlock(
            1, kernel_size=1, stride=1,
            kernel_initializer=tf.constant_initializer(1 / 5))  # TODO

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
        block_4_pre_dense = self.pre_dense_4(layers.add([
            block_4_pre_dense_256, block_3_down]))
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        # Block 5
        block_5_pre_dense_512 = self.pre_dense_5_0(block_4_pre_dense_256)
        block_5_pre_dense = self.pre_dense_5(layers.add([
            block_5_pre_dense_512, block_4_down]))
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
        block_cat = layers.concatenate(scales)  # BxHxWX6
        block_cat = self.block_cat(block_cat)  # BxHxWX1

        outputs = [layers.Activation('sigmoid', name='scale{}'.format(i))(out)
                   for i, out in enumerate(scales)] + \
                  [layers.Activation('sigmoid', name='fused')(block_cat)]

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return [input_shape[:-1] + (1,) for _ in range(7)]

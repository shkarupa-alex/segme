import tensorflow as tf
from tensorflow.keras import Model, initializers, layers, utils
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')

    @shape_type_conversion
    def build(self, input_shape):
        self.prep = layers.Lambda(
            lambda img: preprocess_input(tf.cast(img, tf.float32), 'channels_last', 'caffe'), name='preprocess')

        self.block_1 = DoubleConvBlock(32, 64, stride=2)
        self.block_2 = DoubleConvBlock(128, activation='linear')
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
        self.pre_dense_6_0 = SingleConvBlock(256)
        self.pre_dense_6 = SingleConvBlock(256)

        self.up_block_1 = UpConvBlock(1)
        self.up_block_2 = UpConvBlock(1)
        self.up_block_3 = UpConvBlock(2)
        self.up_block_4 = UpConvBlock(3)
        self.up_block_5 = UpConvBlock(4)
        self.up_block_6 = UpConvBlock(4)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.prep(inputs)

        # Block 1
        conv1_2 = self.block_1(x)
        rconv1 = self.side_1(conv1_2)
        output1 = self.up_block_1(conv1_2)

        # Block 2
        block2_xcp = self.block_2(conv1_2)
        maxpool2_1 = self.maxpool(block2_xcp)
        add2_1 = layers.add([maxpool2_1, rconv1])
        rconv2 = self.side_2(add2_1)
        output2 = self.up_block_2(block2_xcp)

        # Block 3
        addb2_4b3 = self.pre_dense_3(maxpool2_1)
        block3_xcp = self.dblock_3([add2_1, addb2_4b3])
        maxpool3_1 = self.maxpool(block3_xcp)
        add3_1 = layers.add([maxpool3_1, rconv2])
        rconv3 = self.side_3(add3_1)
        output3 = self.up_block_3(block3_xcp)

        # Block 4
        conv_b2b4 = self.pre_dense_2(maxpool2_1)
        addb2b3 = layers.add([conv_b2b4, maxpool3_1])
        addb3_4b4 = self.pre_dense_4(addb2b3)
        block4_xcp = self.dblock_4([add3_1, addb3_4b4])
        maxpool4_1 = self.maxpool(block4_xcp)
        add4_1 = layers.add([maxpool4_1, rconv3])
        rconv4 = self.side_4(add4_1)
        output4 = self.up_block_4(block4_xcp)

        # Block 5
        convb3_2ab4 = self.pre_dense_5_0(conv_b2b4)
        addb2b5 = layers.add([convb3_2ab4, maxpool4_1])
        addb2b5 = self.pre_dense_5(addb2b5)
        block5_xcp = self.dblock_5([add4_1, addb2b5])
        add5_1 = layers.add([block5_xcp, rconv4])
        output5 = self.up_block_5(block5_xcp)

        # Block 6
        block6_xcp_pre = self.pre_dense_6_0(add5_1)
        addb25_2b6 = self.pre_dense_6(block5_xcp)
        block6_xcp = self.dblock_6([block6_xcp_pre, addb25_2b6])
        output6 = self.up_block_6(block6_xcp)

        # concatenate multiscale outputs
        scales = [output1, output2, output3, output4, output5, output6]
        outputs = layers.concatenate(scales)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (6,)


def build_dexi_ned(channels):
    inputs = layers.Input(name='image', shape=[None, None, channels], dtype='uint8')
    outputs = DexiNed()(inputs)
    outputs = ClassificationHead(1, kernel_initializer=initializers.constant(1 / 6))(outputs)
    model = Model(inputs=inputs, outputs=outputs, name='dexi_ned')

    return model

from keras import backend, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from . import core
from . import port


@register_keras_serializable(package='SegMe')
class Backbone(layers.Layer):
    _scales = [1, 2, 4, 8, 16, 32]
    _config = {
        # Backbones and layers to take features in the following order:
        # (x1, x2, x4, x8, x16, x32) - `x4` mean that features has 4 times less
        # spatial resolution (height & width) than input image. Also known as
        # "output stride".

        # ======================================================================
        #                   applications
        # ======================================================================

        # MobileNet
        'mobilenet': (core.MobileNet, (
            # sm
            None, 'conv_pw_1_relu', 'conv_pw_3_relu',
            'conv_pw_5_relu', 'conv_pw_11_relu', 'conv_pw_13_relu'
        )),
        'mobilenet_dw': (core.MobileNet, (
            None, 'conv_dw_1_relu', 'conv_dw_3_relu',
            'conv_dw_5_relu', 'conv_dw_11_relu', 'conv_dw_13_relu'
        )),
        'mobilenet_v2': (core.MobileNetV2, (
            # sm
            None, 'block_1_expand_relu', 'block_3_expand_relu',
            'block_6_expand_relu', 'block_13_expand_relu', 'out_relu'
        )),
        'mobilenet_v2_dw': (core.MobileNetV2, (
            # deeplab
            None, 'expanded_conv_depthwise_relu', 'block_2_depthwise_relu',
            'block_5_depthwise_relu', 'block_12_depthwise_relu',
            'block_16_depthwise_relu'  # block_16_project_BN in deeplab
        )),

        # DenseNet
        'densenet121': (core.DenseNet121, (
            # sm
            None, 'conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'
        )),
        'densenet169': (core.DenseNet169, (
            # sm
            None, 'conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'
        )),
        'densenet201': (core.DenseNet201, (
            # sm
            None, 'conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'
        )),

        # EfficientNet
        'efficientnet_b0': (core.EfficientNetB0, (
            None, 'block2a_expand_activation', 'block3a_expand_activation',
            'block4a_expand_activation', 'block6a_expand_activation',
            'top_activation'
        )),
        'efficientnet_b0_dw': (core.EfficientNetB0, (
            None, 'block1a_activation', 'block2b_activation',
            'block3b_activation', 'block5c_activation', 'block7a_activation'
        )),
        'efficientnet_b1': (core.EfficientNetB1, (
            None, 'block2a_expand_activation', 'block3a_expand_activation',
            'block4a_expand_activation', 'block6a_expand_activation',
            'top_activation'
        )),
        'efficientnet_b1_dw': (core.EfficientNetB1, (
            None, 'block1b_activation', 'block2c_activation',
            'block3c_activation', 'block5d_activation', 'block7b_activation'
        )),
        'efficientnet_b2': (core.EfficientNetB2, (
            None, 'block2a_expand_activation', 'block3a_expand_activation',
            'block4a_expand_activation', 'block6a_expand_activation',
            'top_activation'
        )),
        'efficientnet_b2_dw': (core.EfficientNetB2, (
            None, 'block1b_activation', 'block2c_activation',
            'block3c_activation', 'block5d_activation', 'block7b_activation'
        )),
        'efficientnet_b3': (core.EfficientNetB3, (
            None, 'block2a_expand_activation', 'block3a_expand_activation',
            'block4a_expand_activation', 'block6a_expand_activation',
            'top_activation'
        )),
        'efficientnet_b3_dw': (core.EfficientNetB3, (
            None, 'block1b_activation', 'block2c_activation',
            'block3c_activation', 'block5e_activation', 'block7b_activation'
        )),
        'efficientnet_b4': (core.EfficientNetB4, (
            None, 'block2a_expand_activation', 'block3a_expand_activation',
            'block4a_expand_activation', 'block6a_expand_activation',
            'top_activation'
        )),
        'efficientnet_b4_dw': (core.EfficientNetB4, (
            None, 'block1b_activation', 'block2d_activation',
            'block3d_activation', 'block5f_activation', 'block7b_activation'
        )),
        'efficientnet_b5': (core.EfficientNetB5, (
            None, 'block2a_expand_activation', 'block3a_expand_activation',
            'block4a_expand_activation', 'block6a_expand_activation',
            'top_activation'
        )),
        'efficientnet_b5_dw': (core.EfficientNetB5, (
            None, 'block1c_activation', 'block2e_activation',
            'block3e_activation', 'block5g_activation', 'block7c_activation'
        )),
        'efficientnet_b6': (core.EfficientNetB6, (
            None, 'block2a_expand_activation', 'block3a_expand_activation',
            'block4a_expand_activation', 'block6a_expand_activation',
            'top_activation'
        )),
        'efficientnet_b6_dw': (core.EfficientNetB6, (
            None, 'block1c_activation', 'block2f_activation',
            'block3f_activation', 'block5h_activation', 'block7c_activation'
        )),
        'efficientnet_b7': (core.EfficientNetB7, (
            None, 'block2a_expand_activation', 'block3a_expand_activation',
            'block4a_expand_activation', 'block6a_expand_activation',
            'top_activation'
        )),
        'efficientnet_b7_dw': (core.EfficientNetB7, (
            None, 'block1d_activation', 'block2g_activation',
            'block3g_activation', 'block5j_activation', 'block7d_activation'
        )),

        # Inception
        'inception_v3': (core.InceptionV3, (
            # None, 'activation_2', 'activation_4', 'mixed2', 'mixed7',
            # 'mixed10'
            None, 9, 16, 86, 228, 310
        )),  # scales shape mismatch
        'inception_resnet_v2': (core.InceptionResNetV2, (
            # None, 'activation_2', 'activation_4', 'block35_10_ac',
            # 'block17_20_ac', 'conv_7b_ac'
            None, 9, 16, 260, 594, 779
        )),  # scales shape mismatch
        'xception': (core.Xception, (
            None, 'block2_sepconv2_bn', 'block3_sepconv2_bn',
            'block4_sepconv2_bn', 'block13_sepconv2_bn', 'block14_sepconv2_act'
        )),  # scales shape mismatch

        # NASNet
        # 'nasnetlarge': unable to find right nodes in tree
        # 'nasnetmobile': unable to find right nodes in tree

        # ResNets
        'resnet_50': (core.ResNet50, (
            None, 'conv1_relu', 'conv2_block3_out', 'conv3_block4_out',
            'conv4_block6_out', 'conv5_block3_out'
        )),
        'resnet_50_mid': (core.ResNet50, (
            # deeplab
            None, 'conv1_relu', 'conv2_block2_3_bn', 'conv3_block3_3_bn',
            'conv4_block5_3_bn', 'conv5_block2_3_bn'
        )),
        'resnet_101': (core.ResNet101, (
            None, 'conv1_relu', 'conv2_block3_out', 'conv3_block4_out',
            'conv4_block23_out', 'conv5_block3_out'
        )),
        'resnet_101_mid': (core.ResNet101, (
            # deeplab
            None, 'conv1_relu', 'conv2_block2_3_bn', 'conv3_block3_3_bn',
            'conv4_block22_3_bn', 'conv5_block2_3_bn'
        )),
        'resnet_152': (core.ResNet152, (
            None, 'conv1_relu', 'conv2_block3_out', 'conv3_block8_out',
            'conv4_block36_out', 'conv5_block3_out'
        )),
        'resnet_152_mid': (core.ResNet152, (
            None, 'conv1_relu', 'conv2_block2_3_bn', 'conv3_block7_3_bn',
            'conv4_block35_3_bn', 'conv5_block2_3_bn'
        )),
        'resnet_50_v2': (core.ResNet50V2, (
            None, 'conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu',
            'conv4_block6_1_relu', 'post_relu'
        )),
        'resnet101v2': (core.ResNet101V2, (
            None, 'conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu',
            'conv4_block23_1_relu', 'post_relu'
        )),
        'resnet52v2': (core.ResNet152V2, (
            None, 'conv1_conv', 'conv2_block3_1_relu', 'conv3_block8_1_relu',
            'conv4_block36_1_relu', 'post_relu'
        )),

        # VGG
        'vgg_16': (core.VGG16, (
            # deeplab, sm
            'block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3',
            'block5_conv3', None
        )),
        'vgg_19': (core.VGG19, (
            # deeplab, sm
            'block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4',
            'block5_conv4', None
        )),

        # ======================================================================
        #                   port
        # ======================================================================
        'aligned_xception_41': (port.AlignedXception41, (
            None, 'entry_flow/block1/unit1/sepconv2_pointwise_bn', 'entry_flow/block2/unit1/sepconv2_pointwise_bn',
            'entry_flow/block3/unit1/sepconv2_pointwise_bn', 'exit_flow/block1/unit1/sepconv2_pointwise_bn',
            'exit_flow/block2/unit1/sepconv3_pointwise_bn'
        )),
        'aligned_xception_41_stride_16': (port.AlignedXception41Stride16, (
            None, 'entry_flow/block1/unit1/sepconv2_pointwise_bn', 'entry_flow/block2/unit1/sepconv2_pointwise_bn',
            'entry_flow/block3/unit1/sepconv2_pointwise_bn', 'exit_flow/block2/unit1/sepconv3_pointwise_bn', None
        )),
        'aligned_xception_41_stride_8': (port.AlignedXception41Stride8, (
            None, 'entry_flow/block1/unit1/sepconv2_pointwise_bn', 'entry_flow/block2/unit1/sepconv2_pointwise_bn',
            'exit_flow/block2/unit1/sepconv3_pointwise_bn', None, None
        )),
        'aligned_xception_65': (port.AlignedXception65, (
            None, 'entry_flow/block1/unit1/sepconv2_pointwise_bn', 'entry_flow/block2/unit1/sepconv2_pointwise_bn',
            'entry_flow/block3/unit1/sepconv2_pointwise_bn', 'exit_flow/block1/unit1/sepconv2_pointwise_bn',
            'exit_flow/block2/unit1/sepconv3_pointwise_bn'
        )),
        'aligned_xception_65_stride_16': (port.AlignedXception65Stride16, (
            None, 'entry_flow/block1/unit1/sepconv2_pointwise_bn', 'entry_flow/block2/unit1/sepconv2_pointwise_bn',
            'entry_flow/block3/unit1/sepconv2_pointwise_bn', 'exit_flow/block2/unit1/sepconv3_pointwise_bn', None
        )),
        'aligned_xception_65_stride_8': (port.AlignedXception65Stride8, (
            None, 'entry_flow/block1/unit1/sepconv2_pointwise_bn', 'entry_flow/block2/unit1/sepconv2_pointwise_bn',
            'exit_flow/block2/unit1/sepconv3_pointwise_bn', None, None
        )),
        'aligned_xception_71': (port.AlignedXception71, (
            None, 'entry_flow/block1/unit1/sepconv2_pointwise_bn', 'entry_flow/block3/unit1/sepconv2_pointwise_bn',
            'entry_flow/block5/unit1/sepconv2_pointwise_bn', 'exit_flow/block1/unit1/sepconv2_pointwise_bn',
            'exit_flow/block2/unit1/sepconv3_pointwise_bn'
        )),
        'aligned_xception_71_stride_16': (port.AlignedXception71Stride16, (
            None, 'entry_flow/block1/unit1/sepconv2_pointwise_bn', 'entry_flow/block3/unit1/sepconv2_pointwise_bn',
            'entry_flow/block5/unit1/sepconv2_pointwise_bn', 'exit_flow/block2/unit1/sepconv3_pointwise_bn', None
        )),
        'aligned_xception_71_stride_8': (port.AlignedXception71Stride8, (
            None, 'entry_flow/block1/unit1/sepconv2_pointwise_bn', 'entry_flow/block3/unit1/sepconv2_pointwise_bn',
            'exit_flow/block2/unit1/sepconv3_pointwise_bn', None, None
        )),

        'bit_s_r50x1': (port.BiT_S_R50x1, (
            None, 'standardized_conv2d', 'block1_out', 'block2_out', 'block3_out', 'block4_out'
        )),
        # Bad weights
        # 'bit_s_r50x3': (port.BiT_S_R50x3, (
        #     None, 'root_block.standardized_conv2d', 'block1', 'block2', 'block3', 'block4'
        # )),
        # 'bit_s_r101x1': (port.BiT_S_R101x1, (
        #     None, 'root_block.standardized_conv2d', 'block1', 'block2', 'block3', 'block4'
        # )),
        'bit_s_r101x3': (port.BiT_S_R101x3, (
            None, 'standardized_conv2d', 'block1_out', 'block2_out', 'block3_out', 'block4_out'
        )),
        'bit_s_r152x4': (port.BiT_S_R152x4, (
            None, 'standardized_conv2d', 'block1_out', 'block2_out', 'block3_out', 'block4_out'
        )),
        'bit_m_r50x1': (port.BiT_M_R50x1, (
            None, 'standardized_conv2d', 'block1_out', 'block2_out', 'block3_out', 'block4_out'
        )),
        'bit_m_r50x3': (port.BiT_M_R50x3, (
            None, 'standardized_conv2d', 'block1_out', 'block2_out', 'block3_out', 'block4_out'
        )),
        'bit_m_r101x1': (port.BiT_M_R101x1, (
            None, 'standardized_conv2d', 'block1_out', 'block2_out', 'block3_out', 'block4_out'
        )),
        'bit_m_r101x3': (port.BiT_M_R101x3, (
            None, 'standardized_conv2d', 'block1_out', 'block2_out', 'block3_out', 'block4_out'
        )),
        'bit_m_r152x4': (port.BiT_M_R152x4, (
            None, 'standardized_conv2d', 'block1_out', 'block2_out', 'block3_out', 'block4_out'
        )),

        'mobilenet_v3_small': (port.MobileNetV3Small, (
            # None, 'activation', 'activation_2', 'activation_6',
            # 'activation_16', 'activation_22'
            None, 4, 20, 38, 117, 165
        )),
        'mobilenet_v3_small_dw': (port.MobileNetV3Small, (
            # deeplab
            # None, 'activation', 'activation_1', 'activation_5',
            # 'activation_15', 'activation_21'
            None, 4, 8, 32, 104, 152
        )),
        'mobilenet_v3_large': (port.MobileNetV3Large, (
            # None, 'activation_2', 'activation_6', 'activation_12',
            # 'activation_24', 'activation_30'
            None, 13, 31, 79, 146, 194
        )),
        'mobilenet_v3_large_dw': (port.MobileNetV3Large, (
            # deeplab
            # None, 'activation_1', 'activation_5', 'activation_11',
            # 'activation_23', 'activation_29'
            None, 7, 25, 66, 133, 181
        )),

        # NASNet
        # 'pnasnet': deeplab/core/nas_network.py,
        # 'nas_hnasnet': deeplab/core/nas_network.py,

        # Wide ResNet, Xception, ResNeXt
    }

    def __init__(self, arch, init, trainable, scales=None, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.input_spec = layers.InputSpec(ndim=4, dtype='uint8')

        if arch not in self._config:
            raise ValueError('Unsupported backbone')

        if init is None and not trainable:
            raise ValueError('Backbone should be trainable if initial weights not provided')

        bad_scales = set(scales or []).difference(self._scales)
        if bad_scales:
            raise ValueError('Unsupported scales: {}'.format(bad_scales))

        self.arch = arch
        self.init = init
        self.scales = scales

    @shape_type_conversion
    def build(self, input_shape):
        if 'channels_last' != backend.image_data_format():
            raise ValueError('Only NHWC mode (channels last) supported')

        channel_size = input_shape[-1]
        if channel_size is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channel_size}, dtype='uint8')

        bone_model, default_feats = self._config[self.arch]

        if self.scales is None:
            use_feats = list(filter(None, default_feats))
        else:
            feats_idx = [self._scales.index(sc) for sc in self.scales]
            use_feats = [default_feats[fi] for fi in feats_idx]
            if None in use_feats:
                bad_idx = [fi for fi, uf in enumerate(use_feats) if uf is None]
                bad_scales = [self.scales[sc] for sc in bad_idx]
                raise ValueError('Some scales are unavailable: {}'.format(bad_scales))

        self.bone = bone_model(channel_size, use_feats, self.init)
        self.bone.trainable = self.trainable

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return self.bone(inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.bone.compute_output_shape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'arch': self.arch,
            'init': self.init,
            'scales': self.scales
        })

        return config

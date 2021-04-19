import tensorflow as tf
from tensorflow.keras import layers, regularizers, utils
from tensorflow.python.keras.utils.tf_utils import shape_type_conversion


@utils.register_keras_serializable(package='SegMe')
class StandardizedConv2D(layers.Conv2D):
    """Implements the abs/1903.10520 technique (see go/dune-gn).

    You can simply replace any Conv2D with this one to use re-parametrized
    convolution operation in which the kernels are standardized before conv.

    Note that it does not come with extra learnable scale/bias parameters,
    as those used in "Weight normalization" (abs/1602.07868). This does not
    matter if combined with BN/GN/..., but it would matter if the convolution
    was used standalone.

    Author: Lucas Beyer
    """

    @shape_type_conversion
    def build(self, input_shape):
        super(StandardizedConv2D, self).build(input_shape)

        # Wrap a standardization around the conv OP.
        default_conv_op = self._convolution_op

        def standardized_conv_op(inputs, kernel):
            # Kernel has shape HWIO, normalize over HWI
            mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
            # Author code uses std + 1e-5
            return default_conv_op(inputs, (kernel - mean) / tf.sqrt(var + 1e-10))

        self._convolution_op = standardized_conv_op
        self.built = True

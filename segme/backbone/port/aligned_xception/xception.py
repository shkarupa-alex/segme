"""Xception model.
"Xception: Deep Learning with Depthwise Separable Convolutions" Francois Chollet
https://arxiv.org/abs/1610.02357

We implement the modified version by Jifeng Dai et al. for their COCO 2017 detection challenge submission, where the
model is made deeper and has aligned features for dense prediction tasks. See their slides for details:
"Deformable Convolutional Networks -- COCO Detection and Segmentation Challenge 2017 Entry"
Haozhi Qi, Zheng Zhang, Bin Xiao, Han Hu, Bowen Cheng, Yichen Wei and Jifeng Dai
ICCV 2017 COCO Challenge workshop http://presentations.cocodataset.org/COCO17-Detect-MSRA.pdf

We made a few more changes on top of MSRA's modifications:
1. Fully convolutional: All the max-pooling layers are replaced with separable conv2d with stride = 2. This allows us
 to use atrous convolution to extract feature maps at any resolution.
2. We support adding ReLU and BatchNorm after depthwise convolution, motivated by the design of MobileNetv1.
 "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
 Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
 https://arxiv.org/abs/1704.04861
"""
from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import layer_utils
from tensorflow.python.lib.io import file_io

layers = VersionAwareLayers()


def fixed_padding(inputs, kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation. Should be a positive integer.
      rate: An integer, rate for atrous convolution.
    Returns:
      output: A tensor of size [batch, height_out, width_out, channels] with the input, either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = layers.ZeroPadding2D((pad_beg, pad_end))(inputs)

    return padded_inputs


def separable_conv2d_same(inputs,
                          num_outputs,
                          kernel_size,
                          stride,
                          activation_fn_in_separable_conv,
                          name):
    """Strided 2-D separable convolution with 'SAME' padding.
    If stride > 1 and use_explicit_padding is True, then we do explicit zero-
    padding, followed by conv2d with 'VALID' padding.
    Note that
       net = separable_conv2d_same(inputs, num_outputs, 3,
         depth_multiplier=1, stride=stride)
    is equivalent to
       net = slim.separable_conv2d(inputs, num_outputs, 3,
         depth_multiplier=1, stride=1, padding='SAME')
       net = resnet_utils.subsample(net, factor=stride)
    whereas
       net = slim.separable_conv2d(inputs, num_outputs, 3, stride=stride,
         depth_multiplier=1, padding='SAME')
    is different when the input's height or width is even, which is why we add the
    current function.
    Consequently, if the input feature map has even height or width, setting
    `use_explicit_padding=False` will result in feature misalignment by one pixel
    along the corresponding dimension.
    Args:
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      num_outputs: An integer, the number of output filters.
      kernel_size: An int with the kernel_size of the filters.
      stride: An integer, the output stride.
      activation_fn_in_separable_conv: Includes activation function in the separable convolution or not.
      name: Scope.
    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    if stride == 1:
        depth_padding = 'same'
    else:
        depth_padding = 'valid'
        inputs = fixed_padding(inputs, kernel_size, 1)

    outputs = inputs

    if not activation_fn_in_separable_conv:
        outputs = layers.ReLU()(outputs)

    outputs = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=stride,
        padding=depth_padding,
        use_bias=False,
        name='{}_depthwise'.format(name))(outputs)
    outputs = layers.BatchNormalization(axis=channel_axis, name='{}_depthwise_bn'.format(name))(outputs)
    if activation_fn_in_separable_conv:
        outputs = layers.ReLU(name='{}_depthwise_relu'.format(name))(outputs)

    outputs = layers.Conv2D(filters=num_outputs, kernel_size=1, name='{}_pointwise'.format(name))(outputs)
    outputs = layers.BatchNormalization(axis=channel_axis, name='{}_pointwise_bn'.format(name))(outputs)
    if activation_fn_in_separable_conv:
        outputs = layers.ReLU(name='{}_pointwise_relu'.format(name))(outputs)

    return outputs


def xception_module(inputs, depth_list, skip_connection_type, stride, activation_fn_in_separable_conv=False, name=''):
    """ An Xception module.
    The output of one Xception module is equal to the sum of `residual` and `shortcut`, where `residual` is the feature
    computed by three separable convolution. The `shortcut` is the feature computed by 1x1 convolution with or without
    striding. In some cases, the `shortcut` path could be a simple identity function or none (i.e, no shortcut).
    Note that we replace the max pooling operations in the Xception module with another separable convolution with
    striding, since atrous rate is not properly supported in current TensorFlow max pooling implementation.
    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth_list: A list of three integers specifying the depth values of one Xception module.
      skip_connection_type: Skip connection type for the residual path. Only supports 'conv', 'sum', or 'none'.
      stride: The block unit's stride. Determines the amount of downsampling of the units output compared to its input.
      activation_fn_in_separable_conv: Includes activation function in the separable convolution or not.
      name: Block scope.
    Returns:
      The Xception module's output.
    """

    if len(depth_list) != 3:
        raise ValueError('Expect three elements in depth_list.')

    residual = inputs
    for i in range(3):
        residual = separable_conv2d_same(
            inputs=residual,
            num_outputs=depth_list[i],
            kernel_size=3,
            stride=stride if i == 2 else 1,
            activation_fn_in_separable_conv=activation_fn_in_separable_conv,
            name='{}/sepconv{}'.format(name, i + 1))

    if skip_connection_type == 'conv':
        # shortcut = conv2d_same(
        #     inputs,
        #     num_outputs=depth_list[-1],
        #     kernel_size=1,
        #     stride=stride,
        #     name='{}_shortcut'.format(name))
        shortcut = layers.Conv2D(
            filters=depth_list[-1],
            kernel_size=1,
            padding='same',
            strides=stride,
            name='{}/shortcut'.format(name))(inputs)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    else:
        raise ValueError('Unsupported skip connection type.')

    return outputs


def xception_block(inputs, depth_list, skip_connection_type, activation_fn_in_separable_conv, num_units, stride, name):
    """ Helper function for creating a Xception block.
    Args:
      inputs: The input tensor.
      depth_list: The depth of the bottleneck layer for each unit.
      skip_connection_type: Skip connection type for the residual path. Only supports 'conv', 'sum', or 'none'.
      activation_fn_in_separable_conv: Includes activation function in the separable convolution or not.
      num_units: The number of units in the block.
      stride: The stride of the block, implemented as a stride in the last unit. All other units have stride=1.
      name: The scope of the block.
    Returns:
      An output tensor.
    """
    outputs = inputs
    for i in range(num_units):
        outputs = xception_module(
            outputs,
            depth_list=depth_list,
            skip_connection_type=skip_connection_type,
            activation_fn_in_separable_conv=activation_fn_in_separable_conv,
            stride=stride,
            name='{}/unit{}'.format(name, i + 1))

    return outputs


def conv2d_same(inputs, num_outputs, kernel_size, stride, name):
    """ Strided 2-D convolution with 'SAME' padding.
    When stride > 1, then we do explicit zero-padding, followed by conv2d with 'VALID' padding.
    Note that
     net = conv2d_same(inputs, num_outputs, 3, stride=stride)
    is equivalent to
     net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
     net = subsample(net, factor=stride)
    whereas
     net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')
    is different when the input's height or width is even, which is why we add the current function.

    Args:
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      num_outputs: An integer, the number of output filters.
      kernel_size: An int with the kernel_size of the filters.
      stride: An integer, the output stride.
      name: Scope.

    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels] with the convolution output.
    """
    if 1 == stride:
        padding = 'same'
    else:
        inputs = fixed_padding(inputs, kernel_size=kernel_size)
        padding = 'valid'

    return layers.Conv2D(
        filters=num_outputs,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        name=name)(inputs)


def xception(
        stack_fn,
        model_name='alignedxception',
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
        **kwargs):
    """ Generator for Xception models.
    This function generates a family of Xception models. See the Xception*() methods for specific model instantiations,
    obtained by selecting different block instantiations that produce Xception of various depths.

    Args:
      stack_fn: a function that returns output tensor for the stacked blocks.
      model_name: string, model name.
      include_top: whether to include the fully-connected layer at the top of the network.
      weights: one of `None` (random initialization) or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
      input_shape: optional shape tuple, only to be specified if `include_top` is False (otherwise the input shape has
        to be `(299, 299, channels)` with `channels_last` data format
      pooling: optional pooling mode for feature extraction when `include_top` is `False`.
        - `None` means that the output of the model will be the 4D tensor output of the last convolutional layer.
        - `avg` means that global average pooling will be applied to the output of the last convolutional layer, and
          thus the output of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
      classes: optional number of classes to classify images into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use on the "top" layer. Ignored unless
        `include_top=True`. Set `classifier_activation=None` to return the logits of the "top" layer.
      **kwargs: For backwards compatibility only.

    Returns:
      A `keras.Model` instance.
    """

    global layers
    if 'layers' in kwargs:
        layers = kwargs.pop('layers')
    else:
        layers = VersionAwareLayers()
    if kwargs:
        raise ValueError('Unknown argument(s): %s' % (kwargs,))
    if not (weights is None or file_io.file_exists(weights)):
        raise ValueError('The `weights` argument should be `None` (random initialization) or the path to the weights '
                         'file to be loaded. Pre-trained weights are unavailable')

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=71,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    x = conv2d_same(img_input, 32, 3, stride=2, name='entry_flow/conv1_1')
    x = layers.BatchNormalization(axis=channel_axis, name='entry_flow/conv1_1_bn')(x)
    x = layers.ReLU(name='entry_flow/conv1_1_relu')(x)

    x = conv2d_same(x, 64, 3, stride=1, name='entry_flow/conv1_2')
    x = layers.BatchNormalization(axis=channel_axis, name='entry_flow/conv1_2_bn')(x)
    x = layers.ReLU(name='entry_flow/conv1_2_relu')(x)

    x = stack_fn(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dropout(0.5)(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, activation=classifier_activation, name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name=model_name)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model


def Xception41(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs):
    """ Xception-41 model """

    def stack_fn(x):
        x = xception_block(
            x,
            depth_list=[128, 128, 128],
            skip_connection_type='conv',
            activation_fn_in_separable_conv=False,
            num_units=1,
            stride=2,
            name='entry_flow/block1')
        x = xception_block(
            x,
            depth_list=[256, 256, 256],
            skip_connection_type='conv',
            activation_fn_in_separable_conv=False,
            num_units=1,
            stride=2,
            name='entry_flow/block2')
        x = xception_block(
            x,
            depth_list=[728, 728, 728],
            skip_connection_type='conv',
            activation_fn_in_separable_conv=False,
            num_units=1,
            stride=2,
            name='entry_flow/block3')
        x = xception_block(
            x,
            depth_list=[728, 728, 728],
            skip_connection_type='sum',
            activation_fn_in_separable_conv=False,
            num_units=8,
            stride=1,
            name='middle_flow/block1')
        x = xception_block(
            x,
            depth_list=[728, 1024, 1024],
            skip_connection_type='conv',
            activation_fn_in_separable_conv=False,
            num_units=1,
            stride=2,
            name='exit_flow/block1')
        x = xception_block(
            x,
            depth_list=[1536, 1536, 2048],
            skip_connection_type='none',
            activation_fn_in_separable_conv=True,
            num_units=1,
            stride=1,
            name='exit_flow/block2')
        return x

    return xception(
        stack_fn,
        model_name='alignedxception41',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        **kwargs)


def Xception65(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs):
    """ Xception-65 model """

    def stack_fn(x):
        x = xception_block(
            x,
            depth_list=[128, 128, 128],
            skip_connection_type='conv',
            activation_fn_in_separable_conv=False,
            num_units=1,
            stride=2,
            name='entry_flow/block1')
        x = xception_block(
            x,
            depth_list=[256, 256, 256],
            skip_connection_type='conv',
            activation_fn_in_separable_conv=False,
            num_units=1,
            stride=2,
            name='entry_flow/block2')
        x = xception_block(
            x,
            depth_list=[728, 728, 728],
            skip_connection_type='conv',
            activation_fn_in_separable_conv=False,
            num_units=1,
            stride=2,
            name='entry_flow/block3')
        x = xception_block(
            x,
            depth_list=[728, 728, 728],
            skip_connection_type='sum',
            activation_fn_in_separable_conv=False,
            num_units=16,
            stride=1,
            name='middle_flow/block1')
        x = xception_block(
            x,
            depth_list=[728, 1024, 1024],
            skip_connection_type='conv',
            activation_fn_in_separable_conv=False,
            num_units=1,
            stride=2,
            name='exit_flow/block1')
        x = xception_block(
            x,
            depth_list=[1536, 1536, 2048],
            skip_connection_type='none',
            activation_fn_in_separable_conv=True,
            num_units=1,
            stride=1,
            name='exit_flow/block2')
        return x

    return xception(
        stack_fn,
        model_name='alignedxception65',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        **kwargs)


def Xception71(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        **kwargs):
    """ Xception-71 model """

    def stack_fn(x):
        x = xception_block(
            x,
            depth_list=[128, 128, 128],
            skip_connection_type='conv',
            activation_fn_in_separable_conv=False,
            num_units=1,
            stride=2,
            name='entry_flow/block1')
        x = xception_block(
            x,
            depth_list=[256, 256, 256],
            skip_connection_type='conv',
            activation_fn_in_separable_conv=False,
            num_units=1,
            stride=1,
            name='entry_flow/block2')
        x = xception_block(
            x,
            depth_list=[256, 256, 256],
            skip_connection_type='conv',
            activation_fn_in_separable_conv=False,
            num_units=1,
            stride=2,
            name='entry_flow/block3')
        x = xception_block(
            x,
            depth_list=[728, 728, 728],
            skip_connection_type='conv',
            activation_fn_in_separable_conv=False,
            num_units=1,
            stride=1,
            name='entry_flow/block4')
        x = xception_block(
            x,
            depth_list=[728, 728, 728],
            skip_connection_type='conv',
            activation_fn_in_separable_conv=False,
            num_units=1,
            stride=2,
            name='entry_flow/block5')
        x = xception_block(
            x,
            depth_list=[728, 728, 728],
            skip_connection_type='sum',
            activation_fn_in_separable_conv=False,
            num_units=16,
            stride=1,
            name='middle_flow/block1')
        x = xception_block(
            x,
            depth_list=[728, 1024, 1024],
            skip_connection_type='conv',
            activation_fn_in_separable_conv=False,
            num_units=1,
            stride=2,
            name='exit_flow/block1')
        x = xception_block(
            x,
            depth_list=[1536, 1536, 2048],
            skip_connection_type='none',
            activation_fn_in_separable_conv=True,
            num_units=1,
            stride=1,
            name='exit_flow/block2')
        return x

    return xception(
        stack_fn,
        model_name='alignedxception71',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        **kwargs)


def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode='tf')


def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode='',
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TF,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__

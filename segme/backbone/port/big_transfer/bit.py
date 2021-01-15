from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import keras_export
from .models import ResnetV2, NUM_UNITS

BASE_WEIGHTS_PATH = 'https://storage.googleapis.com/bit_models/'
WEIGHTS_HASHES = {
    'BiT-S-R50x1': '6cf1d1e2713303309e519bcaeff83ff9',
    'BiT-S-R50x3': '6cf1d1e2713303309e519bcaeff83ff9',
    'BiT-S-R101x1': 'e997faeec447e05d5e398303c233e4d3',
    'BiT-S-R101x3': 'f58452086fd56b204a9538c50277e10d',
    'BiT-S-R152x4': 'e997faeec447e05d5e398303c233e4d3',
    'BiT-M-R50x1': '482c5f2671bf6dad5e458260bfd6ac51',
    'BiT-M-R50x3': '215158ea03f1fa71907c5d7b266617a7',
    'BiT-M-R101x1': '8af399a1e846dbcce4001210240b805b',
    'BiT-M-R101x3': '5063875d671620bb49be6885ec23868c',
    'BiT-M-R152x4': '',
}


def BiT(model_name,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        **kwargs):
    """Instantiates the BiT-ResNet architecture.

    Reference:
    - [Big Transfer (BiT): General Visual Representation Learning](
        https://arxiv.org/abs/1912.11370)

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Arguments:
      model_name: string, model name.
      layer at the top of the network.
      weights: one of `None` (random initialization),
        'imagenet' (pre-training on ImageNet),
        or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)` (with `channels_last` data format)
        or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels.
      **kwargs: For backwards compatibility only.
    Returns:
      A `keras.Model` instance.

    Raises:
      ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
    """
    global layers
    if 'layers' in kwargs:
        layers = kwargs.pop('layers')
    else:
        layers = VersionAwareLayers()
    kwargs.pop('include_top')
    if kwargs:
        raise ValueError('Unknown argument(s): %s' % (kwargs,))
    if not (weights in {'imagenet', None} or file_io.file_exists_v2(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet) '
                         'or the path to the weights file to be loaded.')

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=False,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Create base model.
    resnet = ResnetV2(
        num_units=NUM_UNITS[model_name],
        num_outputs=21843 if '-M-' in model_name else 1000,
        filters_factor=int(model_name[-1]) * 4,
        name='resnet')
    x = resnet(img_input)

    # Load weights.
    if model_name in WEIGHTS_HASHES:
        file_name = model_name + '.h5'
        file_hash = WEIGHTS_HASHES[model_name]
        weights_path = data_utils.get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir='models',
            file_hash=file_hash)
        resnet.load_weights(weights_path)
    elif weights is not None:
        resnet.load_weights(weights)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name=model_name)

    return model


def BiT_S_R50x1(weights='imagenet', input_tensor=None, input_shape=None, **kwargs):
    return BiT('BiT-S-R50x1', weights=weights, input_tensor=input_tensor, input_shape=input_shape, **kwargs)


def BiT_S_R50x3(weights='imagenet', input_tensor=None, input_shape=None, **kwargs):
    return BiT('BiT-S-R50x3', weights=weights, input_tensor=input_tensor, input_shape=input_shape, **kwargs)


def BiT_S_R101x1(weights='imagenet', input_tensor=None, input_shape=None, **kwargs):
    return BiT('BiT-S-R101x1', weights=weights, input_tensor=input_tensor, input_shape=input_shape, **kwargs)


def BiT_S_R101x3(weights='imagenet', input_tensor=None, input_shape=None, **kwargs):
    return BiT('BiT-S-R101x3', weights=weights, input_tensor=input_tensor, input_shape=input_shape, **kwargs)


def BiT_S_R152x4(weights='imagenet', input_tensor=None, input_shape=None, **kwargs):
    return BiT('BiT-S-R152x4', weights=weights, input_tensor=input_tensor, input_shape=input_shape, **kwargs)


def BiT_M_R50x1(weights='imagenet', input_tensor=None, input_shape=None, **kwargs):
    return BiT('BiT-M-R50x1', weights=weights, input_tensor=input_tensor, input_shape=input_shape, **kwargs)


def BiT_M_R50x3(weights='imagenet', input_tensor=None, input_shape=None, **kwargs):
    return BiT('BiT-M-R50x3', weights=weights, input_tensor=input_tensor, input_shape=input_shape, **kwargs)


def BiT_M_R101x1(weights='imagenet', input_tensor=None, input_shape=None, **kwargs):
    return BiT('BiT-M-R101x1', weights=weights, input_tensor=input_tensor, input_shape=input_shape, **kwargs)


def BiT_M_R101x3(weights='imagenet', input_tensor=None, input_shape=None, **kwargs):
    return BiT('BiT-M-R101x3', weights=weights, input_tensor=input_tensor, input_shape=input_shape, **kwargs)


def BiT_M_R152x4(weights='imagenet', input_tensor=None, input_shape=None, **kwargs):
    return BiT('BiT-M-R152x4', weights=weights, input_tensor=input_tensor, input_shape=input_shape, **kwargs)


def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode='tf')

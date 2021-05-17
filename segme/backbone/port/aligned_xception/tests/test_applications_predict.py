import numpy as np
from absl.testing import parameterized
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.platform import test
from ..xception import Xception41, Xception65, Xception71, preprocess_input, decode_predictions

MODEL_LIST = [Xception41, Xception65, Xception71]

TEST_IMAGE_PATH = ('https://storage.googleapis.com/tensorflow/'
                   'keras-applications/tests/elephant.jpg')
_IMAGENET_CLASSES = 1000


class ApplicationsLoadWeightTest(test.TestCase, parameterized.TestCase):
    @parameterized.parameters(*MODEL_LIST)
    def test_application_predict_odd(self, app):
        model = app(input_shape=(224, 224, 3))
        _assert_shape_equal(model.output_shape, (None, _IMAGENET_CLASSES))
        x = _get_elephant((224, 224))
        x = preprocess_input(x)
        preds = model.predict(x)
        decode_predictions(preds)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_predict_even(self, app):
        model = app(input_shape=(299, 299, 3))
        _assert_shape_equal(model.output_shape, (None, _IMAGENET_CLASSES))
        x = _get_elephant((299, 299))
        x = preprocess_input(x)
        preds = model.predict(x)
        decode_predictions(preds)


def _get_elephant(target_size):
    # For models that don't include a Flatten step,
    # the default is to accept variable-size inputs
    # even when loading ImageNet weights (since it is possible).
    # In this case, default to 299x299.
    if target_size[0] is None:
        target_size = (299, 299)
    test_image = data_utils.get_file('elephant.jpg', TEST_IMAGE_PATH)
    img = image.load_img(test_image, target_size=tuple(target_size))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def _assert_shape_equal(shape1, shape2):
    if len(shape1) != len(shape2):
        raise AssertionError(
            'Shapes are different rank: %s vs %s' % (shape1, shape2))
    if shape1 != shape2:
        raise AssertionError('Shapes differ: %s vs %s' % (shape1, shape2))


if __name__ == '__main__':
    test.main()

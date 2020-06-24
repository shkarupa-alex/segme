# import numpy as np
# from absl.testing import parameterized
# from tensorflow.python.keras.preprocessing import image
# from tensorflow.python.keras.utils import data_utils
# from tensorflow.python.platform import test
# from ..mobilenet_v3 import MobileNetV3Small, MobileNetV3Large
#
# MODEL_LIST = [
#     (mobilenet_v3, [MobileNetV3Small, MobileNetV3Large])
# ]
#
# TEST_IMAGE_PATH = ('https://storage.googleapis.com/tensorflow/'
#                    'keras-applications/tests/elephant.jpg')
# _IMAGENET_CLASSES = 1000
#
#
# class ApplicationsLoadWeightTest(test.TestCase, parameterized.TestCase):
#     @parameterized.parameters(*MODEL_LIST)
#     def test_application_pretrained_weights_loading(self, app_module, apps):
#         for app in apps:
#             model = app(weights='imagenet')
#             _assert_shape_equal(model.output_shape, (None, _IMAGENET_CLASSES))
#             x = _get_elephant(model.input_shape[1:3])
#             x = app_module.preprocess_input(x)
#             preds = model.predict(x)
#             names = [p[1] for p in app_module.decode_predictions(preds)[0]]
#             # Test correct label is in top 3 (weak correctness test).
#             self.assertIn('African_elephant', names[:3])
#
#
# def _get_elephant(target_size):
#     # For models that don't include a Flatten step,
#     # the default is to accept variable-size inputs
#     # even when loading ImageNet weights (since it is possible).
#     # In this case, default to 299x299.
#     if target_size[0] is None:
#         target_size = (299, 299)
#     test_image = data_utils.get_file('elephant.jpg', TEST_IMAGE_PATH)
#     img = image.load_img(test_image, target_size=tuple(target_size))
#     x = image.img_to_array(img)
#     return np.expand_dims(x, axis=0)
#
#
# def _assert_shape_equal(shape1, shape2):
#     if len(shape1) != len(shape2):
#         raise AssertionError(
#             'Shapes are different rank: %s vs %s' % (shape1, shape2))
#     if shape1 != shape2:
#         raise AssertionError('Shapes differ: %s vs %s' % (shape1, shape2))
#
#
# if __name__ == '__main__':
#     test.main()

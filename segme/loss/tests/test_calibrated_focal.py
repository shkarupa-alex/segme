# import numpy as np
# import tensorflow as tf
# from tensorflow.python.keras import keras_parameterized
# from ..calibrated_focal import SigmoidCalibratedFocalCrossEntropy
# from ..calibrated_focal import sigmoid_calibrated_focal_crossentropy
#
#
# def _to_logit(prob):
#     logit = np.log(prob / (1.0 - prob))
#
#     return logit
#
#
# def _log10(x):
#     numerator = tf.math.log(x)
#     denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
#
#     return numerator / denominator
#
#
# @keras_parameterized.run_all_keras_modes
# class TestDeepLabV3Plus(keras_parameterized.TestCase):
#     def test_config(self):
#         bce_obj = SigmoidCalibratedFocalCrossEntropy(
#             reduction=tf.keras.losses.Reduction.NONE,
#             name='sigmoid_calibrated_focal_crossentropy'
#         )
#         self.assertEqual(bce_obj.name, 'sigmoid_calibrated_focal_crossentropy')
#         self.assertEqual(bce_obj.reduction, tf.keras.losses.Reduction.NONE)
#
#     def test_value_gamma_2(self):
#         prediction_tensor = tf.constant([
#             [_to_logit(0.97)],
#             [_to_logit(0.45)],
#             [_to_logit(0.03)],
#         ], tf.float32)
#         target_tensor = tf.constant([[1], [1], [0]], tf.float32)
#
#         fl = sigmoid_calibrated_focal_crossentropy(
#             y_true=target_tensor,
#             y_pred=prediction_tensor,
#             from_logits=True,
#             gamma0=2.,
#             gamma1=2.
#         )
#         fl = self.evaluate(fl).tolist()
#
#         self.assertAllClose(fl, [6.8533022e-06, 6.0387149e-02, 2.0559946e-05])
#
#     def test_keras_model_compile(self):
#         model = tf.keras.models.Sequential([
#             tf.keras.layers.Input(shape=(100,)),
#             tf.keras.layers.Dense(5, activation='softmax')]
#         )
#         model.compile(loss='SegMe>sigmoid_calibrated_focal_crossentropy')
#
#     def test_gamma_default(self):
#         prediction_tensor = tf.constant([[0.97], [0.45], [0.03]], tf.float32)
#         target_tensor = tf.constant([[1], [1], [0]], tf.float32)
#
#         fl = sigmoid_calibrated_focal_crossentropy(
#             y_true=target_tensor, y_pred=prediction_tensor)
#         fl = self.evaluate(fl).tolist()
#
#         self.assertAllClose(
#             fl, [2.0559804e-07, 3.3212923e-02, 5.5511362e-10], atol=1e-8)
#
#
# if __name__ == '__main__':
#     tf.test.main()

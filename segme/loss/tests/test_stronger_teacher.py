import numpy as np
import tensorflow as tf
from keras.src import layers
from keras.src import models
from keras.src import testing

from segme.loss.stronger_teacher import StrongerTeacherLoss
from segme.loss.stronger_teacher import stronger_teacher_loss
from segme.loss.tests.test_common_loss import MULTI_LOGITS
from segme.loss.tests.test_common_loss import MULTI_TARGETS
from segme.loss.tests.test_common_loss import MULTI_WEIGHTS


class TestStrongerTeacherLoss(testing.TestCase):
    def test_config(self):
        loss = StrongerTeacherLoss(reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros(self):
        logits = -10.0 * tf.one_hot(
            tf.zeros((3, 8, 8), "int32"), 2, dtype="float32"
        )
        targets = logits

        result = stronger_teacher_loss(
            y_true=targets, y_pred=logits, sample_weight=None, temperature=1.0
        )

        self.assertAlmostEqual(result, -1.0, decimal=3)

    def test_ones(self):
        logits = 10.0 * tf.one_hot(
            tf.zeros((3, 8, 8), "int32"), 2, dtype="float32"
        )
        targets = logits

        result = stronger_teacher_loss(
            y_true=targets, y_pred=logits, sample_weight=None, temperature=1.0
        )

        self.assertAlmostEqual(result, -1.0, decimal=3)

    def test_false(self):
        logits = -10.0 * tf.one_hot(
            tf.zeros((3, 8, 8), "int32"), 2, dtype="float32"
        )
        targets = tf.reverse(logits, axis=[-1])

        result = stronger_teacher_loss(
            y_true=targets, y_pred=logits, sample_weight=None, temperature=1.0
        )

        self.assertAlmostEqual(result, 1.0, decimal=3)

    def test_true(self):
        logits = 10.0 * tf.one_hot(
            tf.zeros((3, 8, 8), "int32"), 2, dtype="float32"
        )
        targets = tf.reverse(logits, axis=[-1])

        result = stronger_teacher_loss(
            y_true=targets, y_pred=logits, sample_weight=None, temperature=1.0
        )

        self.assertAlmostEqual(result, 1.0, decimal=3)

    def test_value(self):
        targets = tf.one_hot(
            tf.squeeze(MULTI_TARGETS, axis=-1), MULTI_LOGITS.shape[-1]
        )
        loss = StrongerTeacherLoss()
        result = loss(targets, MULTI_LOGITS)

        # Original: -0.2666435 (1.7333565 - 2) (difference in cosine
        # similarity normalization)
        self.assertAlmostEqual(result, -0.2088526, decimal=6)

    def test_weight(self):
        targets = tf.one_hot(
            tf.squeeze(MULTI_TARGETS, axis=-1), MULTI_LOGITS.shape[-1]
        )

        loss = StrongerTeacherLoss()

        result = loss(targets[:, :, :1], MULTI_LOGITS[:, :, :1])
        self.assertAlmostEqual(result, 0.06347078, decimal=5)

        result = loss(targets, MULTI_LOGITS, MULTI_WEIGHTS)
        self.assertAlmostEqual(result, -0.12937176, decimal=5)

        result = loss(targets, MULTI_LOGITS, MULTI_WEIGHTS * 2.0)
        self.assertAlmostEqual(result, -0.12937176, decimal=5)

    # Not applicable due to intra-class loss part
    # def test_batch(self):
    #     probs = np.random.rand(2, 224, 224, 2).astype('float32')
    #     targets = np.random.rand(2, 224, 224, 2).astype('float32')
    #
    #     loss = StrongerTeacherLoss()
    #     result0 = loss(targets, probs)
    #     result1 = sum([
    #       loss(targets[i:i + 1], probs[i:i + 1]) for i in range(2)]) / 2
    #
    #     self.assertAlmostEqual(result0, result1, decimal=6)

    def test_model(self):
        model = models.Sequential([layers.Dense(10, activation="linear")])
        model.compile(
            loss="SegMe>Loss>StrongerTeacherLoss",
        )
        model.fit(np.zeros((2, 8, 8, 1)), np.zeros((2, 8, 8, 10), "float32"))
        models.Sequential.from_config(model.get_config())

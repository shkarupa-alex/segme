import numpy as np
import tensorflow as tf
from keras.src import layers
from keras.src import models
from keras.src import testing

from segme.loss.clip_foundation import ClipFoundationLoss
from segme.loss.clip_foundation import clip_foundation_loss
from segme.loss.tests.test_common_loss import MULTI_LOGITS
from segme.loss.tests.test_common_loss import MULTI_TARGETS
from segme.loss.tests.test_common_loss import MULTI_WEIGHTS


class TestClipFoundationLoss(testing.TestCase):
    def test_config(self):
        loss = ClipFoundationLoss(reduction="none", name="loss1")
        self.assertEqual(loss.name, "loss1")
        self.assertEqual(loss.reduction, "none")

    def test_zeros(self):
        logits = -10.0 * tf.one_hot(
            tf.zeros((3, 8, 8), "int32"), 2, dtype="float32"
        )
        targets = tf.concat([logits, logits], axis=-1)

        result = clip_foundation_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            scale=100.0,
            bias=None,
            temperature=(1.0, 1.0, 1.0),
            weight=(1.0, 1.0, 1.0),
        )

        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_ones(self):
        logits = 10.0 * tf.one_hot(
            tf.zeros((3, 8, 8), "int32"), 2, dtype="float32"
        )
        targets = tf.concat([logits, logits], axis=-1)

        result = clip_foundation_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            scale=100.0,
            bias=None,
            temperature=(1.0, 1.0, 1.0),
            weight=(1.0, 1.0, 1.0),
        )

        self.assertAllClose(result, [0.0] * 3, atol=6e-3)

    def test_false(self):
        logits = tf.reshape(
            tf.range(3 * 8 * 8 * 2, dtype="float32") / (3 * 8 * 8 * 2),
            [3, 8, 8, 2],
        )
        targets = tf.reverse(logits, axis=[0])
        targets = tf.concat([targets, targets], axis=-1)

        result = clip_foundation_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            scale=100.0,
            bias=None,
            temperature=(1.0, 1.0, 1.0),
            weight=(1.0, 1.0, 1.0),
        )

        self.assertAllClose(
            result, [2.3200176, 0.49343127, 2.7184358], atol=6e-3
        )

    def test_true(self):
        logits = tf.reshape(
            tf.range(3 * 8 * 8 * 2, dtype="float32") / (3 * 8 * 8 * 2),
            [3, 8, 8, 2],
        )
        targets = tf.reverse(logits, axis=[-1])
        targets = tf.concat([targets, targets], axis=-1)

        result = clip_foundation_loss(
            y_true=targets,
            y_pred=logits,
            sample_weight=None,
            scale=100.0,
            bias=None,
            temperature=(1.0, 1.0, 1.0),
            weight=(1.0, 1.0, 1.0),
        )

        self.assertAllClose(result, [4.758, 0.0, 0.0], atol=6e-3)

    def test_value(self):
        targets = tf.concat(
            [
                tf.transpose(MULTI_LOGITS, [0, 3, 2, 1]),
                tf.reshape(
                    tf.transpose(MULTI_LOGITS, [0, 2, 1, 3]), MULTI_LOGITS.shape
                ),
            ],
            axis=-1,
        )

        loss = ClipFoundationLoss()
        result = loss(targets, MULTI_LOGITS)

        self.assertAlmostEqual(result, 94.09963, decimal=5)

    def test_weight(self):
        targets = tf.concat(
            [
                tf.transpose(MULTI_LOGITS, [0, 3, 2, 1]),
                tf.reshape(
                    tf.transpose(MULTI_LOGITS, [0, 2, 1, 3]), MULTI_LOGITS.shape
                ),
            ],
            axis=-1,
        )

        loss = ClipFoundationLoss()

        result = loss(targets[:, :, :1], MULTI_LOGITS[:, :, :1])
        self.assertAlmostEqual(result, 107.85203, decimal=5)

        result = loss(targets, MULTI_LOGITS, MULTI_WEIGHTS)
        self.assertAlmostEqual(result, 130.91214, decimal=5)

        result = loss(targets, MULTI_LOGITS, MULTI_WEIGHTS * 2.0)
        self.assertAlmostEqual(result, 130.91214 * 2, decimal=5)

    # Not applicable due to batch-to-batch loss nature
    # def test_batch(self):
    #     probs = np.random.rand(2, 224, 224, 2).astype('float32')
    #     targets = np.random.rand(2, 224, 224, 2).astype('float32')
    #
    #     loss = ClipFoundationLoss()
    #     result0 = loss(targets, probs)
    #     result1 = sum([loss(targets[i:i + 1], probs[i:i + 1]) for i in range(2)]) / 2
    #
    #     self.assertAlmostEqual(result0, result1, decimal=6)

    def test_model(self):
        model = models.Sequential([layers.Dense(4, activation="linear")])
        model.compile(
            loss="SegMe>Loss>ClipFoundationLoss",
            
        )
        model.fit(np.zeros((2, 8, 8, 4)), np.zeros((2, 8, 8, 8), "float32"))
        models.Sequential.from_config(model.get_config())

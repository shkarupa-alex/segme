import numpy as np
from keras.src import losses
from keras.src import testing
from keras.src.applications import mobilenet_v3

from segme.common.distill import FeatureDistillation
from segme.common.distill import KullbackLeibler
from segme.common.distill import StrongerTeacher


class TestFeatureDistillation(testing.TestCase):
    def test_model(self):
        teacher = mobilenet_v3.MobileNetV3Large(include_top=False, weights=None)
        student = mobilenet_v3.MobileNetV3Small(include_top=False, weights=None)
        model = FeatureDistillation(
            student, teacher, jit_compile_teacher=True, jit_compile=True
        )

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype("float32")
        labels = np.random.uniform(size=(32,)).round().astype("int32")
        model.fit(x=inputs, y=labels, batch_size=4)


class TestKullbackLeibler(testing.TestCase):
    def test_model(self):
        teacher = mobilenet_v3.MobileNetV3Large(
            classifier_activation="linear", weights=None
        )
        student = mobilenet_v3.MobileNetV3Small(
            classifier_activation="linear", weights=None
        )
        model = KullbackLeibler(
            student,
            teacher,
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            loss_weights=1.0,
            metrics="accuracy",
            weighted_metrics="sparse_top_k_categorical_accuracy",
            jit_compile_teacher=True,
            jit_compile=True,
        )

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype("float32")
        labels = np.random.uniform(size=(32,)).round().astype("int32")
        model.fit(x=inputs, y=labels, batch_size=4)


class TestStrongerTeacher(testing.TestCase):
    def test_model(self):
        teacher = mobilenet_v3.MobileNetV3Large(
            classifier_activation="linear", weights=None
        )
        student = mobilenet_v3.MobileNetV3Small(
            classifier_activation="linear", weights=None
        )
        model = StrongerTeacher(
            student,
            teacher,
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            loss_weights=None,
            metrics="accuracy",
            weighted_metrics="sparse_top_k_categorical_accuracy",
            jit_compile_teacher=True,
            jit_compile=True,
        )

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype("float32")
        labels = np.random.uniform(size=(32,)).round().astype("int32")
        model.fit(x=inputs, y=labels, batch_size=4)

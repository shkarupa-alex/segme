import tempfile

import numpy as np
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.applications import mobilenet_v3
from keras.src.dtype_policies import dtype_policy

from segme.common.distill import FeatureDistillation
from segme.common.distill import KullbackLeibler
from segme.common.distill import StrongerTeacher


class TestFeatureDistillation(testing.TestCase):
    def setUp(self):
        self.default_policy = dtype_policy.dtype_policy()
        super().setUp()

    def tearDown(self):
        dtype_policy.set_dtype_policy(self.default_policy)
        super().tearDown()

    def test_model(self):
        teacher = mobilenet_v3.MobileNetV3Large(
            include_top=False, include_preprocessing=False
        )

        student = mobilenet_v3.MobileNetV3Small(
            include_top=False, include_preprocessing=False
        )
        student_proj = layers.Conv2D(teacher.outputs[0].shape[-1], 1)(
            student.outputs[0]
        )
        student = FeatureDistillation(
            inputs=student.inputs, outputs=student_proj
        )
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype("float32")
        labels = np.random.uniform(size=(32,)).round().astype("int32")
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)

    def test_model_fp16(self):
        dtype_policy.set_dtype_policy("mixed_float16")

        teacher = mobilenet_v3.MobileNetV3Large(
            include_top=False, include_preprocessing=False
        )

        student = mobilenet_v3.MobileNetV3Small(
            include_top=False, include_preprocessing=False
        )
        student_proj = layers.Conv2D(teacher.outputs[0].shape[-1], 1)(
            student.outputs[0]
        )
        student = FeatureDistillation(
            inputs=student.inputs, outputs=student_proj
        )
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype("float16")
        labels = np.random.uniform(size=(32,)).round().astype("int32")
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)

    def test_serializable(self):
        teacher = mobilenet_v3.MobileNetV3Large(
            include_top=False, include_preprocessing=False
        )

        student = mobilenet_v3.MobileNetV3Small(
            include_top=False, include_preprocessing=False
        )
        student_proj = layers.Conv2D(teacher.outputs[0].shape[-1], 1)(
            student.outputs[0]
        )
        student = FeatureDistillation(
            inputs=student.inputs, outputs=student_proj
        )
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype("float32")
        labels = np.random.uniform(size=(32,)).round().astype("int32")
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)
        loss0 = student.evaluate(x=inputs, y=labels, batch_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            student.save(tmpdir)
            del student
            del student_proj
            backend.clear_session()
            restored = models.load_model(tmpdir)

        restored.set_teacher(teacher)
        loss1 = restored.evaluate(x=inputs, y=labels, batch_size=4)

        self.assertEqual(loss0, loss1)


class TestKullbackLeibler(testing.TestCase):

    def test_model(self):
        teacher = mobilenet_v3.MobileNetV3Large(
            classifier_activation="linear", include_preprocessing=False
        )

        student = mobilenet_v3.MobileNetV3Small(
            classifier_activation="linear", include_preprocessing=False
        )
        student = KullbackLeibler(
            inputs=student.inputs, outputs=student.outputs
        )
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype("float32")
        labels = np.random.uniform(size=(32,)).round().astype("int32")
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)

    def test_model_fp16(self):
        dtype_policy.set_dtype_policy("mixed_float16")

        teacher = mobilenet_v3.MobileNetV3Large(
            classifier_activation="linear", include_preprocessing=False
        )

        student = mobilenet_v3.MobileNetV3Small(
            classifier_activation="linear", include_preprocessing=False
        )
        student = KullbackLeibler(
            inputs=student.inputs, outputs=student.outputs
        )
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype("float16")
        labels = np.random.uniform(size=(32,)).round().astype("int32")
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)

    def test_serializable(self):
        teacher = mobilenet_v3.MobileNetV3Large(
            classifier_activation="linear", include_preprocessing=False
        )

        student = mobilenet_v3.MobileNetV3Small(
            classifier_activation="linear", include_preprocessing=False
        )
        student = KullbackLeibler(
            inputs=student.inputs, outputs=student.outputs
        )
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype("float32")
        labels = np.random.uniform(size=(32,)).round().astype("int32")
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)
        loss0 = student.evaluate(x=inputs, y=labels, batch_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            student.save(tmpdir)
            del student
            backend.clear_session()
            restored = models.load_model(tmpdir)

        restored.set_teacher(teacher)
        loss1 = restored.evaluate(x=inputs, y=labels, batch_size=4)

        self.assertEqual(loss0, loss1)


class TestStrongerTeacher(testing.TestCase):

    def test_model(self):
        teacher = mobilenet_v3.MobileNetV3Large(
            classifier_activation="linear", include_preprocessing=False
        )

        student = mobilenet_v3.MobileNetV3Small(
            classifier_activation="linear", include_preprocessing=False
        )
        student = StrongerTeacher(
            inputs=student.inputs, outputs=student.outputs
        )
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype("float32")
        labels = np.random.uniform(size=(32,)).round().astype("int32")
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)

    def test_model_fp16(self):
        dtype_policy.set_dtype_policy("mixed_float16")

        teacher = mobilenet_v3.MobileNetV3Large(
            classifier_activation="linear", include_preprocessing=False
        )

        student = mobilenet_v3.MobileNetV3Small(
            classifier_activation="linear", include_preprocessing=False
        )
        student = StrongerTeacher(
            inputs=student.inputs, outputs=student.outputs
        )
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype("float16")
        labels = np.random.uniform(size=(32,)).round().astype("int32")
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)

    def test_serializable(self):
        teacher = mobilenet_v3.MobileNetV3Large(
            classifier_activation="linear", include_preprocessing=False
        )

        student = mobilenet_v3.MobileNetV3Small(
            classifier_activation="linear", include_preprocessing=False
        )
        student = StrongerTeacher(
            inputs=student.inputs, outputs=student.outputs
        )
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype("float32")
        labels = np.random.uniform(size=(32,)).round().astype("int32")
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)
        loss0 = student.evaluate(x=inputs, y=labels, batch_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            student.save(tmpdir)
            del student
            backend.clear_session()
            restored = models.load_model(tmpdir)

        restored.set_teacher(teacher)
        loss1 = restored.evaluate(x=inputs, y=labels, batch_size=4)

        self.assertEqual(loss0, loss1)

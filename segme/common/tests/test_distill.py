import numpy as np
import tempfile
import tensorflow as tf
from tf_keras import applications, backend, layers, mixed_precision, models
from tf_keras.src.testing_infra import test_combinations
from segme.common.distill import FeatureDistillation, KullbackLeibler, StrongerTeacher


@test_combinations.run_all_keras_modes
class TestFeatureDistillation(test_combinations.TestCase):
    def setUp(self):
        super(TestFeatureDistillation, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestFeatureDistillation, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_model(self):
        teacher = applications.MobileNetV3Large(include_top=False, include_preprocessing=False)

        student = applications.MobileNetV3Small(include_top=False, include_preprocessing=False)
        student_proj = layers.Conv2D(teacher.outputs[0].shape[-1], 1)(student.outputs[0])
        student = FeatureDistillation(inputs=student.inputs, outputs=student_proj)
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype('float32')
        labels = np.random.uniform(size=(32,)).round().astype('int32')
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)

    def test_model_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')

        teacher = applications.MobileNetV3Large(include_top=False, include_preprocessing=False)

        student = applications.MobileNetV3Small(include_top=False, include_preprocessing=False)
        student_proj = layers.Conv2D(teacher.outputs[0].shape[-1], 1)(student.outputs[0])
        student = FeatureDistillation(inputs=student.inputs, outputs=student_proj)
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype('float16')
        labels = np.random.uniform(size=(32,)).round().astype('int32')
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)

    def test_serializable(self):
        teacher = applications.MobileNetV3Large(include_top=False, include_preprocessing=False)

        student = applications.MobileNetV3Small(include_top=False, include_preprocessing=False)
        student_proj = layers.Conv2D(teacher.outputs[0].shape[-1], 1)(student.outputs[0])
        student = FeatureDistillation(inputs=student.inputs, outputs=student_proj)
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype('float32')
        labels = np.random.uniform(size=(32,)).round().astype('int32')
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


@test_combinations.run_all_keras_modes
class TestKullbackLeibler(test_combinations.TestCase):
    def setUp(self):
        super(TestKullbackLeibler, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestKullbackLeibler, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_model(self):
        teacher = applications.MobileNetV3Large(classifier_activation='linear', include_preprocessing=False)

        student = applications.MobileNetV3Small(classifier_activation='linear', include_preprocessing=False)
        student = KullbackLeibler(inputs=student.inputs, outputs=student.outputs)
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype('float32')
        labels = np.random.uniform(size=(32,)).round().astype('int32')
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)

    def test_model_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')

        teacher = applications.MobileNetV3Large(classifier_activation='linear', include_preprocessing=False)

        student = applications.MobileNetV3Small(classifier_activation='linear', include_preprocessing=False)
        student = KullbackLeibler(inputs=student.inputs, outputs=student.outputs)
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype('float16')
        labels = np.random.uniform(size=(32,)).round().astype('int32')
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)

    def test_serializable(self):
        teacher = applications.MobileNetV3Large(classifier_activation='linear', include_preprocessing=False)

        student = applications.MobileNetV3Small(classifier_activation='linear', include_preprocessing=False)
        student = KullbackLeibler(inputs=student.inputs, outputs=student.outputs)
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype('float32')
        labels = np.random.uniform(size=(32,)).round().astype('int32')
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


@test_combinations.run_all_keras_modes
class TestStrongerTeacher(test_combinations.TestCase):
    def setUp(self):
        super(TestStrongerTeacher, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestStrongerTeacher, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_model(self):
        teacher = applications.MobileNetV3Large(classifier_activation='linear', include_preprocessing=False)

        student = applications.MobileNetV3Small(classifier_activation='linear', include_preprocessing=False)
        student = StrongerTeacher(inputs=student.inputs, outputs=student.outputs)
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype('float32')
        labels = np.random.uniform(size=(32,)).round().astype('int32')
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)

    def test_model_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')

        teacher = applications.MobileNetV3Large(classifier_activation='linear', include_preprocessing=False)

        student = applications.MobileNetV3Small(classifier_activation='linear', include_preprocessing=False)
        student = StrongerTeacher(inputs=student.inputs, outputs=student.outputs)
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype('float16')
        labels = np.random.uniform(size=(32,)).round().astype('int32')
        student.compile()
        student.fit(x=inputs, y=labels, batch_size=4)

    def test_serializable(self):
        teacher = applications.MobileNetV3Large(classifier_activation='linear', include_preprocessing=False)

        student = applications.MobileNetV3Small(classifier_activation='linear', include_preprocessing=False)
        student = StrongerTeacher(inputs=student.inputs, outputs=student.outputs)
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 224, 224, 3)).astype('float32')
        labels = np.random.uniform(size=(32,)).round().astype('int32')
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


if __name__ == '__main__':
    tf.test.main()

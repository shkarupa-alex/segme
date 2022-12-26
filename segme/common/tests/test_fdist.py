import numpy as np
import tempfile
import tensorflow as tf
from keras import applications, backend, layers, models
from keras.mixed_precision import policy as mixed_precision
from keras.testing_infra import test_combinations
from segme.common.fdist import FeatureDistillation


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

        inputs = np.random.uniform(size=(32, 256, 256, 3)).astype('float32')
        student.compile()
        student.fit(x=inputs, batch_size=4)

    def test_model_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')

        teacher = applications.MobileNetV3Large(include_top=False, include_preprocessing=False)

        student = applications.MobileNetV3Small(include_top=False, include_preprocessing=False)
        student_proj = layers.Conv2D(teacher.outputs[0].shape[-1], 1)(student.outputs[0])
        student = FeatureDistillation(inputs=student.inputs, outputs=student_proj)
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 256, 256, 3)).astype('float16')
        student.compile()
        student.fit(x=inputs, batch_size=4)

    def test_serializable(self):
        teacher = applications.MobileNetV3Large(include_top=False, include_preprocessing=False)

        student = applications.MobileNetV3Small(include_top=False, include_preprocessing=False)
        student_proj = layers.Conv2D(teacher.outputs[0].shape[-1], 1)(student.outputs[0])
        student = FeatureDistillation(inputs=student.inputs, outputs=student_proj)
        student.set_teacher(teacher)

        inputs = np.random.uniform(size=(32, 256, 256, 3)).astype('float32')
        student.compile()
        student.fit(x=inputs, batch_size=4)
        loss0 = student.evaluate(x=inputs, batch_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            student.save(tmpdir)
            del student
            del student_proj
            backend.clear_session()
            restored = models.load_model(tmpdir)

        restored.set_teacher(teacher)
        loss1 = restored.evaluate(x=inputs, batch_size=4)

        self.assertEqual(loss0, loss1)


if __name__ == '__main__':
    tf.test.main()

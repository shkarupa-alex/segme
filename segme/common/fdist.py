import tensorflow as tf
from keras import layers, models
from keras.saving import register_keras_serializable
from keras.src.engine.data_adapter import unpack_x_y_sample_weight
from tensorflow.python.framework import convert_to_constants
from segme.loss.soft_mae import SoftMeanAbsoluteError


@register_keras_serializable(package='SegMe>Common')
class FeatureDistillation(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = None
        self.whitener = layers.LayerNormalization(center=False, scale=False, epsilon=1.001e-5, dtype='float32')

    def set_teacher(self, model, jit_compile=None):
        if not isinstance(model, models.Model):
            raise ValueError(f'Expecting teacher model to be an instance of keras.Model. Got: {type(model)}')

        teacher_sp = tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
        teacher_fn = tf.function(
            lambda x: self.whitener(model(x, training=False)),
            jit_compile=jit_compile, reduce_retracing=True)
        teacher_fn = teacher_fn.get_concrete_function(teacher_sp)
        teacher_fn = convert_to_constants.convert_variables_to_constants_v2(teacher_fn)
        self.teacher = teacher_fn

    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None,
                run_eagerly=None, steps_per_execution=None, jit_compile=None, **kwargs):
        if loss is not None or metrics is not None or loss_weights is not None or weighted_metrics is not None:
            raise ValueError('User defined loss, loss_weights, metrics and weighted_metrics not supported.')
        loss = SoftMeanAbsoluteError(beta=2., name='soft_mae_loss')

        super().compile(
            optimizer=optimizer, loss=loss, metrics=None, loss_weights=None, weighted_metrics=None,
            run_eagerly=run_eagerly, steps_per_execution=steps_per_execution, jit_compile=jit_compile, **kwargs)

    def _get_compile_args(self, user_metrics=True):
        compile_args = super()._get_compile_args(user_metrics)
        compile_args['loss'] = None
        compile_args['loss_weights'] = None

        return compile_args

    def train_step(self, data):
        if self.teacher is None:
            raise ValueError('Expecting teacher model to be set through "set_teacher" method.')

        x, y, sample_weight = unpack_x_y_sample_weight(data)
        del y, sample_weight

        if isinstance(x, dict) and 'teacher' in x and 'student' in x:
            xt, xs = x['teacher'], x['student']
        else:
            xt, xs = x, x

        yt = self.teacher(xt)

        with tf.GradientTape() as tape:
            y_pred = self(xs, training=True)
            y_pred = self.whitener(y_pred)
            loss = self.compute_loss(xs, yt, y_pred)

        self._validate_target_and_loss(yt, loss)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        return self.compute_metrics(xs, yt, y_pred, None)

    def test_step(self, data):
        if self.teacher is None:
            raise ValueError('Expecting teacher model to be set through "set_teacher" method.')

        x, y, sample_weight = unpack_x_y_sample_weight(data)
        del y, sample_weight

        if isinstance(x, dict) and 'teacher' in x and 'student' in x:
            xt, xs = x['teacher'], x['student']
        else:
            xt, xs = x, x

        yt = self.teacher(xt)

        y_pred = self(xs, training=False)
        y_pred = self.whitener(y_pred)
        self.compute_loss(xs, yt, y_pred)

        return self.compute_metrics(xs, yt, y_pred, None)

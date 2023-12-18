import tensorflow as tf
from keras import layers, models
from keras.saving import register_keras_serializable
from keras.src.engine.data_adapter import unpack_x_y_sample_weight
from tensorflow.python.framework import convert_to_constants
from segme.loss.cross_entropy import CrossEntropyLoss
from segme.loss.kl_divergence import KLDivergenceLoss
from segme.loss.soft_mae import SoftMeanAbsoluteError
from segme.loss.stronger_teacher import StrongerTeacherLoss


@register_keras_serializable(package='SegMe>Common')
class ModelDistillation(models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_fn = None
        self.dist_prep = self.distillation_preprocessor()
        self.targ_prep = self.target_preprocessor()
        self.dist_loss = self.distillation_loss()
        self.targ_loss = self.target_loss()

    def distillation_preprocessor(self):
        return layers.Activation('linear', dtype='float32', name='distill_head_prep')

    def target_preprocessor(self):
        raise NotImplementedError

    def distillation_loss(self):
        raise NotImplementedError

    def target_loss(self):
        raise NotImplementedError

    def set_teacher(self, model, jit_compile=None):
        if not isinstance(model, models.Model):
            raise ValueError(f'Expecting teacher model to be an instance of keras.Model. Got: {type(model)}')

        teacher_sp = [tf.TensorSpec(i.shape, i.dtype) for i in model.inputs]
        teacher_sp = teacher_sp[0] if 1 == len(teacher_sp) else teacher_sp
        teacher_fn = tf.function(
            lambda x: self.dist_prep(model(x, training=False)),
            jit_compile=jit_compile, reduce_retracing=True)
        teacher_fn = teacher_fn.get_concrete_function(teacher_sp)
        teacher_fn = convert_to_constants.convert_variables_to_constants_v2(teacher_fn)
        self.teacher_fn = teacher_fn

    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None,
                run_eagerly=None, steps_per_execution=None, jit_compile=None, **kwargs):
        if loss is not None or loss_weights is not None:
            raise ValueError('User defined `loss` and `loss_weights` not supported.')

        if self.targ_loss:
            loss = [self.dist_loss, self.targ_loss]
            if metrics is not None:
                metrics = [None, metrics]
            if weighted_metrics is not None:
                weighted_metrics = [None, weighted_metrics]
        else:
            if metrics is not None or weighted_metrics is not None:
                raise ValueError('User defined `metrics` and `weighted_metrics` not supported.')
            loss = self.dist_loss

        super().compile(
            optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=None, weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly, steps_per_execution=steps_per_execution, jit_compile=jit_compile, **kwargs)

    def _get_compile_args(self, user_metrics=True):
        compile_args = super()._get_compile_args(user_metrics)
        compile_args['loss'] = None
        compile_args['loss_weights'] = None

        return compile_args

    def train_step(self, data):
        if self.teacher_fn is None:
            raise ValueError('Expecting teacher model to be set through "set_teacher" method.')

        x, y, sample_weight = unpack_x_y_sample_weight(data)

        if isinstance(x, dict) and 'teacher' in x and 'student' in x:
            xt, xs = x['teacher'], x['student']
        else:
            xt, xs = x, x

        yt = self.teacher_fn(xt)
        y = [yt, y] if self.targ_loss else yt
        sample_weight = [None, sample_weight] if self.targ_loss else None

        with tf.GradientTape() as tape:
            y_pred = self(xs, training=True)

            if self.targ_loss:
                y_pred = [self.dist_prep(y_pred), self.targ_prep(y_pred)]
            else:
                y_pred = self.dist_prep(y_pred)

            loss = self.compute_loss(xs, y, y_pred, sample_weight)

        self._validate_target_and_loss(y, loss)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        return self.compute_metrics(xs, y, y_pred, sample_weight)

    def test_step(self, data):
        if self.teacher_fn is None:
            raise ValueError('Expecting teacher model to be set through "set_teacher" method.')

        x, y, sample_weight = unpack_x_y_sample_weight(data)

        if isinstance(x, dict) and 'teacher' in x and 'student' in x:
            xt, xs = x['teacher'], x['student']
        else:
            xt, xs = x, x

        yt = self.teacher_fn(xt)
        y = [yt, y] if self.targ_loss else yt
        sample_weight = [None, sample_weight] if self.targ_loss else None

        y_pred = self(xs, training=False)
        if self.targ_loss:
            y_pred = [self.dist_prep(y_pred), self.targ_prep(y_pred)]
        else:
            y_pred = self.dist_prep(y_pred)

        self.compute_loss(xs, y, y_pred, sample_weight)

        return self.compute_metrics(xs, y, y_pred, sample_weight)


@register_keras_serializable(package='SegMe>Common')
class FeatureDistillation(ModelDistillation):
    """ Proposed in: https://arxiv.org/abs/2205.14141 """

    def distillation_preprocessor(self):
        return layers.LayerNormalization(
            center=False, scale=False, epsilon=1.001e-5, dtype='float32', name='distill_head_prep')

    def target_preprocessor(self):
        return None

    def distillation_loss(self):
        return SoftMeanAbsoluteError(beta=2.)

    def target_loss(self):
        return None


@register_keras_serializable(package='SegMe>Common')
class KullbackLeibler(ModelDistillation):
    def distillation_preprocessor(self):
        return layers.Activation('linear', dtype='float32', name='distill_head_prep')

    def target_preprocessor(self):
        return layers.Activation('linear', dtype='float32', name='target_head_prep')

    def distillation_loss(self):
        return KLDivergenceLoss()

    def target_loss(self):
        # TODO: label smoothing
        return CrossEntropyLoss(from_logits=True)


@register_keras_serializable(package='SegMe>Common')
class StrongerTeacher(ModelDistillation):
    """ Proposed in: https://arxiv.org/abs/2205.10536 """
    def distillation_preprocessor(self):
        return layers.Activation('linear', dtype='float32', name='distill_head_prep')

    def target_preprocessor(self):
        return layers.Activation('linear', dtype='float32', name='target_head_prep')

    def distillation_loss(self):
        # TODO: *2
        return StrongerTeacherLoss()

    def target_loss(self):
        # TODO: label smoothing
        return CrossEntropyLoss(from_logits=True)


@register_keras_serializable(package='SegMe>Common')
class ClipFoundation(ModelDistillation):
    """ Proposed in: https://arxiv.org/abs/2303.18232 """
    pass

    # def feature_preprocessor(self):
    #     raise layers.LayerNormalization(center=False, scale=False, epsilon=1.001e-5, dtype='float32')
    #
    # def distillation_loss(self):
    #     return SoftMeanAbsoluteError(beta=2., name='soft_mae_loss')

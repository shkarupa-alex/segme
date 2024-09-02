from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src.saving import register_keras_serializable

from segme.backend import model_inference_fn
from segme.loss.kl_divergence import KLDivergenceLoss
from segme.loss.soft_mae import SoftMeanAbsoluteError
from segme.loss.stronger_teacher import StrongerTeacherLoss


@register_keras_serializable(package="SegMe>Common")
class ModelDistillation(models.Functional):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = None

    def compile(
        self,
        teacher,
        jit_compile_teacher="auto",
        optimizer="rmsprop",
        loss=None,
        loss_weights=None,
        metrics=None,
        weighted_metrics=None,
        run_eagerly=False,
        steps_per_execution=1,
        jit_compile="auto",
        auto_scale_loss=True,
    ):
        self.teacher = model_inference_fn(teacher, jit_compile_teacher)

        if not isinstance(loss, dict):
            raise ValueError(
                f"Loss should be a dict with `student` and `teacher` keys. "
                f"Got: {type(loss)}"
            )
        if {"student", "teacher"}.difference(loss.keys()):
            raise ValueError(
                f"Loss should be a dict with `student` and `teacher` keys. "
                f"Got: {loss.keys()}"
            )

        if loss_weights is not None and not isinstance(loss_weights, dict):
            raise ValueError(
                f"Loss weights should be a dict with `student` and `teacher` "
                f"keys. Got: {type(loss_weights)}"
            )
        if loss_weights is not None and {"student", "teacher"}.difference(
            loss_weights.keys()
        ):
            raise ValueError(
                f"Loss weights should be a dict with `student` and `teacher` "
                f"keys. Got: {loss_weights.keys()}"
            )

        if metrics is not None and not isinstance(metrics, dict):
            raise ValueError(
                f"Metrics should be a dict with `student` key. "
                f"Got: {type(metrics)}"
            )
        if metrics is not None and {"student"}.difference(metrics.keys()):
            raise ValueError(
                f"Metrics should be a dict with `student` key. "
                f"Got: {metrics.keys()}"
            )

        if weighted_metrics is not None and not isinstance(
            weighted_metrics, dict
        ):
            raise ValueError(
                f"Weighted metrics should be a dict with `student` key. "
                f"Got: {type(weighted_metrics)}"
            )
        if weighted_metrics is not None and {"student"}.difference(
            weighted_metrics.keys()
        ):
            raise ValueError(
                f"Weighted metrics should be a dict with `student` key. "
                f"Got: {weighted_metrics.keys()}"
            )

        super().compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            auto_scale_loss=auto_scale_loss,
        )

    def __call__(self, x, training=None):
        if isinstance(x, dict) and "student" in x:
            xs = x["student"]
        else:
            xs = x

        y_pred = super().__call__(xs, training=training)
        if not isinstance(y_pred, dict):
            raise ValueError(
                f"Student model should return a dict with `student` and "
                f"`teacher` keys. Got: {type(y_pred)}"
            )
        if {"student", "teacher"}.difference(y_pred.keys()):
            raise ValueError(
                f"Student model should return a dict with `student` and "
                f"`teacher` keys. Got: {y_pred.keys()}"
            )

        return y_pred

    def compute_loss(
        self,
        x=None,
        y=None,
        y_pred=None,
        sample_weight=None,
        training=True,
    ):
        if self.teacher is None:
            raise ValueError(
                "Expecting teacher model to be set through `compile` method."
            )

        if isinstance(x, dict) and "teacher" in x:
            xt = x["teacher"]
        else:
            xt = x

        yt = self.teacher(xt)
        if isinstance(yt, (list, tuple)):
            if 1 != len(yt):
                raise ValueError("Unexpected teacher output.")
            yt = yt[0]
        yt = ops.stop_gradient(yt)
        y = {
            "student": ops.cast(y, "float32"),
            "teacher": ops.cast(yt, "float32"),
        }

        if not isinstance(y_pred, dict):
            raise ValueError(
                f"Student model should return a dict with `student` and "
                f"`teacher` keys. Got: {type(y_pred)}"
            )
        if {"student", "teacher"}.difference(y_pred.keys()):
            raise ValueError(
                f"Student model should return a dict with `student` and "
                f"`teacher` keys. Got: {y_pred.keys()}"
            )

        if sample_weight is not None and not isinstance(sample_weight, dict):
            sample_weight = {"student": sample_weight, "teacher": None}
        if sample_weight is not None and {"student", "teacher"}.difference(
            sample_weight.keys()
        ):
            raise ValueError(
                f"Sample weights should be a dict with `student` and "
                f"`teacher` keys. Got: {sample_weight.keys()}"
            )

        return super().compute_loss(
            x=x,
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
            training=training,
        )

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        y = {
            "student": ops.cast(y, "float32"),
        }

        if not isinstance(y_pred, dict):
            raise ValueError(
                f"Student model should return a dict with `student` and "
                f"`teacher` keys. Got: {type(y_pred)}"
            )
        if {"student", "teacher"}.difference(y_pred.keys()):
            raise ValueError(
                f"Student model should return a dict with `student` and "
                f"`teacher` keys. Got: {y_pred.keys()}"
            )
        y_pred = {"student": y_pred["student"]}

        if sample_weight is not None and not isinstance(sample_weight, dict):
            sample_weight = {"student": sample_weight}
        if sample_weight is not None and {"student"}.difference(
            sample_weight.keys()
        ):
            raise ValueError(
                f"Sample weights should be a dict with `student` key. "
                f"Got: {sample_weight.keys()}"
            )

        return super().compute_metrics(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
        )


def FeatureDistillation(
    student,
    teacher,
    jit_compile_teacher="auto",
    optimizer="rmsprop",
    run_eagerly=False,
    steps_per_execution=1,
    jit_compile="auto",
    auto_scale_loss=True,
):
    """Proposed in: https://arxiv.org/abs/2205.14141"""
    teacher = models.Model(
        inputs=teacher.inputs,
        outputs=layers.LayerNormalization(
            center=False,
            scale=False,
            epsilon=1.001e-5,
            dtype="float32",
            name="teacher_whiten",
        )(teacher.outputs[0]),
    )

    model = ModelDistillation(
        inputs=student.inputs,
        outputs={
            "student": layers.Activation("linear", name="student")(
                student.outputs[0]
            ),
            "teacher": layers.Conv2D(
                teacher.outputs[0].shape[-1], 1, name="teacher"
            )(student.outputs[0]),
        },
    )
    model.compile(
        teacher,
        jit_compile_teacher=jit_compile_teacher,
        optimizer=optimizer,
        loss={"student": None, "teacher": SoftMeanAbsoluteError(beta=2.0)},
        loss_weights=None,
        metrics=None,
        weighted_metrics=None,
        run_eagerly=run_eagerly,
        steps_per_execution=steps_per_execution,
        jit_compile=jit_compile,
        auto_scale_loss=auto_scale_loss,
    )

    return model


def KullbackLeibler(
    student,
    teacher,
    loss=None,
    loss_weights=None,
    metrics=None,
    weighted_metrics=None,
    jit_compile_teacher="auto",
    optimizer="rmsprop",
    run_eagerly=False,
    steps_per_execution=1,
    jit_compile="auto",
    auto_scale_loss=True,
):
    if loss_weights is None:
        loss_weights = 1.0

    model = ModelDistillation(
        inputs=student.inputs,
        outputs={
            "student": layers.Activation("linear", name="student")(
                student.outputs[0]
            ),
            "teacher": layers.Activation("linear", name="teacher")(
                student.outputs[0]
            ),
        },
    )
    model.compile(
        teacher,
        jit_compile_teacher=jit_compile_teacher,
        optimizer=optimizer,
        loss={"student": loss, "teacher": KLDivergenceLoss()},
        loss_weights={"student": loss_weights, "teacher": 1.0},
        metrics={"student": metrics, "teacher": None},
        weighted_metrics={"student": weighted_metrics, "teacher": None},
        run_eagerly=run_eagerly,
        steps_per_execution=steps_per_execution,
        jit_compile=jit_compile,
        auto_scale_loss=auto_scale_loss,
    )

    return model


def StrongerTeacher(
    student,
    teacher,
    loss=None,
    loss_weights=None,
    metrics=None,
    weighted_metrics=None,
    jit_compile_teacher="auto",
    optimizer="rmsprop",
    run_eagerly=False,
    steps_per_execution=1,
    jit_compile="auto",
    auto_scale_loss=True,
):
    """Proposed in: https://arxiv.org/abs/2205.10536"""

    if loss_weights is None:
        loss_weights = 1.0

    model = ModelDistillation(
        inputs=student.inputs,
        outputs={
            "student": layers.Activation("linear", name="student")(
                student.outputs[0]
            ),
            "teacher": layers.Activation("linear", name="teacher")(
                student.outputs[0]
            ),
        },
    )
    model.compile(
        teacher,
        jit_compile_teacher=jit_compile_teacher,
        optimizer=optimizer,
        loss={"student": loss, "teacher": StrongerTeacherLoss()},
        loss_weights={"student": loss_weights, "teacher": 2.0},
        metrics={"student": metrics, "teacher": None},
        weighted_metrics={"student": weighted_metrics, "teacher": None},
        run_eagerly=run_eagerly,
        steps_per_execution=steps_per_execution,
        jit_compile=jit_compile,
        auto_scale_loss=auto_scale_loss,
    )

    return model

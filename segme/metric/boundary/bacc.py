import cv2
from keras.src import backend
from keras.src import ops
from keras.src.metrics import BinaryAccuracy
from keras.src.metrics import SparseCategoricalAccuracy
from keras.src.saving import register_keras_serializable

from segme.ops import dilation_2d
from segme.ops import erosion_2d


@register_keras_serializable(package="SegMe>Metric>Boundary")
class BinaryBoundaryAccuracy(BinaryAccuracy):
    def __init__(
        self,
        radius=1,
        threshold=0.5,
        name="binary_boundary_accuracy",
        dtype=None,
    ):
        """Creates an `Accuracy` metric instance estimated only in `radius`
        pixels from boundary.

        Args:
            radius: (Optional) int radius of boundary
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name, threshold=threshold, dtype=dtype)
        self.radius = radius
        self.strict = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_weight = boundary_weight(
            y_true, self.radius, self.strict, sample_weight
        )

        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({"radius": self.radius})

        return config


@register_keras_serializable(package="SegMe>Metric>Boundary")
class BinaryApproximateBoundaryAccuracy(BinaryBoundaryAccuracy):
    def __init__(
        self,
        radius=1,
        threshold=0.5,
        name="binary_approximate_boundary_accuracy",
        dtype=None,
    ):
        """Creates an `Accuracy` metric instance estimated only in `radius`
        pixels from boundary.
        Approximating 3x3 ellipse kernel with square one.

        Args:
            radius: (Optional) int radius of boundary
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(
            radius=radius, threshold=threshold, name=name, dtype=dtype
        )
        self.strict = False


@register_keras_serializable(package="SegMe>Metric>Boundary")
class SparseCategoricalBoundaryAccuracy(SparseCategoricalAccuracy):
    def __init__(
        self, radius=1, name="sparse_categorical_boundary_accuracy", dtype=None
    ):
        """Creates a `SparseCategoricalAccuracy` metric instance estimated
        only in `radius` pixels from boundary.

        Args:
            radius: (Optional) int radius of boundary
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(name, dtype=dtype)
        self.radius = radius
        self.strict = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_1h = ops.one_hot(
            ops.squeeze(y_true, -1), y_pred.shape[-1], dtype="int32"
        )
        sample_weight = boundary_weight(
            y_true_1h, self.radius, self.strict, sample_weight
        )

        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({"radius": self.radius})

        return config


@register_keras_serializable(package="SegMe>Metric>Boundary")
class SparseCategoricalApproximateBoundaryAccuracy(
    SparseCategoricalBoundaryAccuracy
):
    def __init__(
        self,
        radius=1,
        name="sparse_categorical_approximate_boundary_accuracy",
        dtype=None,
    ):
        """Creates a `SparseCategoricalAccuracy` metric instance estimated
        only in `radius` pixels from boundary.
        Approximating 3x3 ellipse kernel with square one.

        Args:
            radius: (Optional) int radius of boundary
            name: (Optional) string name of the metric instance.
            dtype: (Optional) data type of the metric result.
        """
        super().__init__(radius=radius, name=name, dtype=dtype)
        self.radius = radius
        self.strict = False


def boundary_weight(y_true, radius, strict, sample_weight):
    if 4 != ops.ndim(y_true):
        raise ValueError("Labels must have rank 4.")

    if strict:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))[..., None]
        kernel = backend.convert_to_tensor(kernel, "int32")
        kernel = ops.tile(kernel, (1, 1, y_true.shape[-1]))

        eroded = dilated = ops.cast(y_true, "int32")
        for _ in range(radius):
            eroded = erosion_2d(eroded, kernel, 1, "same", 1)
            dilated = dilation_2d(dilated, kernel, 1, "same", 1)

        weight = ops.cast(ops.equal(eroded + dilated, 1), "float32")
    else:
        foreground = ops.cast(y_true, "float32")
        for _ in range(radius):
            foreground = ops.average_pool(
                foreground, 3, strides=1, padding="same"
            )

        weight = ops.cast((foreground > 0.0) & (foreground < 1.0), "float32")

    weight = ops.max(weight, axis=-1, keepdims=True)

    batch, height, width, _ = ops.shape(y_true)
    frame = ops.zeros(
        (batch, height - radius * 2, width - radius * 2, 1), "float32"
    )
    frame = ops.pad(
        frame,
        [[0, 0], [radius, radius], [radius, radius], [0, 0]],
        constant_values=1.0,
    )
    weight = ops.maximum(weight, frame)

    if sample_weight is None:
        sample_weight = weight
    else:
        sample_weight *= weight

    return sample_weight

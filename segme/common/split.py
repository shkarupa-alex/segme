from keras.src import layers
from keras.src import ops
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="SegMe>Common")
class Split(layers.Layer):
    def __init__(self, indices_or_sections, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        if not isinstance(indices_or_sections, (int, list, tuple)):
            raise ValueError(
                f"Expected type of `indices_or_sections` to be `int`, `list` "
                f"or `tuple`. Got {type(indices_or_sections)}"
            )

        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def build(self, input_shape):
        dimension = input_shape[self.axis]
        if dimension is not None:
            if (
                isinstance(self.indices_or_sections, int)
                and dimension % self.indices_or_sections
            ):
                raise ValueError(
                    "Split dimension of the inputs should be "
                    "divisible by the number of sections."
                )
            if (
                isinstance(self.indices_or_sections, (list, tuple))
                and max(self.indices_or_sections) > dimension
            ):
                raise ValueError(
                    "Last split section should be inside dimension size"
                )

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return tuple(ops.split(inputs, self.indices_or_sections, self.axis))

    def compute_output_shape(self, input_shape):
        if input_shape[self.axis] is None:
            if isinstance(self.indices_or_sections, int):
                size = self.indices_or_sections
            else:
                size = len(self.indices_or_sections) + 1
            return (input_shape,) * size

        output_shape = list(input_shape)

        if isinstance(self.indices_or_sections, int):
            output_shape[self.axis] = (
                input_shape[self.axis] // self.indices_or_sections
            )

            return (tuple(output_shape),) * self.indices_or_sections

        output_shapes = []
        indices_or_sections = list(self.indices_or_sections)
        for start, stop in zip(
            [0] + indices_or_sections,
            indices_or_sections + [output_shape[self.axis]],
        ):
            output_shape[self.axis] = stop - start
            output_shapes.append(tuple(output_shape))

        return tuple(output_shapes)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"indices_or_sections": self.indices_or_sections, "axis": self.axis}
        )

        return config

from keras.src import backend
from keras.src import layers
from keras.src import ops
from keras.src.saving import register_keras_serializable
from keras.src.utils.argument_validation import standardize_tuple


@register_keras_serializable(package="SegMe>Common")
class SymmetricPadding(layers.ZeroPadding2D):
    def __init__(self, padding=(1, 1), data_format=None, **kwargs):
        super().__init__(padding=padding, data_format=data_format, **kwargs)

        if max(self.padding[0] + self.padding[1]) > 1:
            raise ValueError(
                "Symmetric padding can lead to misbehavior when "
                "padding size > 1"
            )

    def call(self, inputs):
        if self.data_format == "channels_first":
            all_dims_padding = ((0, 0), (0, 0), *self.padding)
        else:
            all_dims_padding = ((0, 0), *self.padding, (0, 0))
        return ops.pad(inputs, all_dims_padding, mode="SYMMETRIC")


def with_divisible_pad(
    op,
    inputs,
    dividers,
    mode="constant",
    constant_values=0,
    dtype=None,
    name=None,
):
    with backend.name_scope(name or "with_divisible_pad"):
        inputs = backend.convert_to_tensor(inputs, dtype)
        if 4 != inputs.shape.rank:
            raise ValueError("Expecting `inputs` rank to be 4.")

        dividers = standardize_tuple(dividers, 2, "standardize_tuple")
        if 1 == max(dividers):
            raise ValueError("Nothing to pad: both multipliers equals to 1.")

        batch, height, width, channels = ops.shape(inputs)
        static_size = isinstance(height, int) and isinstance(width, int)

        h_pad = (dividers[0] - height % dividers[0]) % dividers[0]
        w_pad = (dividers[1] - width % dividers[1]) % dividers[1]
        with_pad = h_pad + w_pad > 0

        hb_pad, wb_pad = h_pad // 2, w_pad // 2
        ha_pad, wa_pad = h_pad - hb_pad, w_pad - wb_pad
        paddings = ((0, 0), (hb_pad, ha_pad), (wb_pad, wa_pad), (0, 0))

        if static_size and with_pad or not static_size:
            outputs = ops.pad(
                inputs, paddings, mode=mode, constant_values=constant_values
            )
        else:
            outputs = inputs

        pad_size = height + h_pad, width + w_pad
        pad_val = hb_pad, ha_pad, wb_pad, wa_pad
        outputs = op(outputs, pad_size=pad_size, pad_val=pad_val)

        if 4 != len(outputs.shape):
            raise ValueError(
                f"Expecting `op` output to have rank 4. "
                f"Got: {len(outputs.shape)}"
            )
        out_batch, out_height, out_width = ops.shape(outputs)[:3]
        if (
            isinstance(batch, int)
            and isinstance(out_batch, int)
            and batch != out_batch
        ):
            raise ValueError(
                "Expecting `op` output batch size to be the same as input one."
            )
        if (
            isinstance(height, int)
            and isinstance(out_height, int)
            and height + h_pad != out_height
        ):
            raise ValueError(
                "Expecting `op` output height to be the same as input one."
            )
        if (
            isinstance(width, int)
            and isinstance(out_width, int)
            and width + w_pad != out_width
        ):
            raise ValueError(
                "Expecting `op` output width to be the same as input one."
            )

        if static_size and with_pad or not static_size:
            outputs = outputs[
                :,
                hb_pad : hb_pad + height,
                wb_pad : wb_pad + width,
            ]

        # TODO: set shape

        return outputs

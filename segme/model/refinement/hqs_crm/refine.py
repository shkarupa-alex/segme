import math

import cv2
import numpy as np
import tensorflow as tf
from tensorflow_hub import KerasLayer


class Refiner:
    def __init__(self, hub_uri):
        self.model = KerasLayer(hub_uri, trainable=False)

        self.image = tf.Variable(
            trainable=False,
            shape=(1, None, None, 3),
            dtype="uint8",
            initial_value=np.zeros((1, 0, 0, 3)).astype("uint8"),
        )
        self.mask = tf.Variable(
            trainable=False,
            shape=(1, None, None, 1),
            dtype="uint8",
            initial_value=np.zeros((1, 0, 0, 1)).astype("uint8"),
        )
        self.coord = tf.Variable(
            trainable=False,
            shape=(1, None, None, 2),
            dtype="float32",
            initial_value=np.zeros((1, 0, 0, 2)).astype("float32"),
        )

    def __call__(self, image, mask, fast=None, max_size=960, up_scale=1):
        if fast is None:
            fast = max(image.shape[:2]) * up_scale <= max_size

        fine = self._global_step(image, mask, max_size)
        if fast:
            return fine

        fine = (fine > 127).astype("uint8") * 255
        fine = self._local_step(image, fine, max_size, up_scale)

        return fine

    def _global_step(self, image, mask, max_size):
        height_width = image.shape[:2]

        if max(height_width) < max_size:
            image = Refiner._resize_max_side(
                image, max_size, cv2.INTER_LANCZOS4
            )
            mask = Refiner._resize_max_side(mask, max_size, cv2.INTER_LANCZOS4)
        elif max(height_width) > max_size:
            image = Refiner._resize_max_side(image, max_size, cv2.INTER_AREA)
            mask = Refiner._resize_max_side(mask, max_size, cv2.INTER_AREA)
        mask = (mask > 127).astype("uint8") * 255

        fine = self._safe_predict(image, mask)

        if max(height_width) < max_size:
            fine = Refiner._resize_fixed_size(
                fine, height_width, interpolation=cv2.INTER_AREA
            )
        elif max(height_width) > max_size:
            fine = Refiner._resize_fixed_size(
                fine, height_width, interpolation=cv2.INTER_LANCZOS4
            )

        return fine

    def _local_step(self, image, mask, max_size, up_scale):
        src_shape = image.shape[:2]

        if up_scale > 1:
            up_size = max(src_shape) * up_scale
            up_size = round(up_size)
            image = Refiner._resize_max_side(image, up_size, cv2.INTER_LANCZOS4)
            mask = Refiner._resize_max_side(mask, up_size, cv2.INTER_LANCZOS4)
            mask = (mask > 127).astype("uint8") * 255

        height, width = mask.shape[:2]
        h_pad = (32 - height % 32) % 32
        w_pad = (32 - width % 32) % 32

        image = np.pad(image, ((0, h_pad), (0, w_pad), (0, 0)))
        mask = np.pad(mask, ((0, h_pad), (0, w_pad)))
        coord = Refiner.make_coord(1, height + h_pad, width + w_pad)
        coord = coord[0, :height, :width]

        fine = np.zeros_like(mask, dtype="uint8")[:height, :width]

        for x_idx in range(math.ceil(width / max_size)):
            for y_idx in range(math.ceil(height / max_size)):
                start_x = x_idx * max_size
                start_y = y_idx * max_size
                end_x = min(width, start_x + max_size)
                end_y = min(height, start_y + max_size)

                # Take crop
                part_coord = coord[start_y:end_y, start_x:end_x]

                # Predict crop
                grid_fine = self._safe_predict(image, mask, part_coord)
                fine[start_y:end_y, start_x:end_x] = grid_fine

        if up_scale > 1:
            fine = Refiner._resize_fixed_size(
                fine, src_shape, interpolation=cv2.INTER_AREA
            )

        return fine

    def _safe_predict(self, image, mask, coord=None):
        if len(image.shape) != 3:
            raise ValueError("Wrong image supplied")
        if image.dtype != "uint8":
            raise ValueError("Wrong image dtype")
        if len(mask.shape) != 2:
            raise ValueError("Wrong mask supplied")
        if mask.dtype != "uint8":
            raise ValueError("Wrong mask dtype")
        if set(np.unique(mask)) - {0, 255}:
            raise ValueError("Wrong mask values")
        if coord is not None and len(coord.shape) != 3:
            raise ValueError("Wrong coord supplied")
        if coord is not None and coord.dtype != "float32":
            raise ValueError("Wrong coord dtype")
        if coord is not None and (coord.min() < -1.0 or coord.max() > 1.0):
            raise ValueError("Wrong coord values")

        height, width = image.shape[:2]
        h_pad = (32 - height % 32) % 32
        w_pad = (32 - width % 32) % 32
        if not (coord is None or 0 == h_pad or 0 == w_pad):
            raise ValueError(
                "Inputs shape should be divisible by 32 if coord provided"
            )

        _image = np.pad(image, ((0, h_pad), (0, w_pad), (0, 0)))
        _mask = np.pad(mask, ((0, h_pad), (0, w_pad)))
        if coord is None:
            _coord = Refiner.make_coord(1, height + h_pad, width + w_pad)
            _coord = _coord[0, :height, :width]
        else:
            _coord = coord

        self.image.assign(_image[None])
        self.mask.assign(_mask[None, ..., None])
        self.coord.assign(_coord[None])

        fine = self.model([self.image, self.mask, self.coord])
        fine = fine.numpy()[0, ..., 0]
        fine = np.round(fine * 255).astype(np.uint8)

        return fine

    @staticmethod
    def _resize_max_side(image, max_size, interpolation=cv2.INTER_LINEAR):
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise ValueError("Wrong image supplied")

        aspect = max_size / max(image.shape[:2])

        return cv2.resize(
            image, (0, 0), fx=aspect, fy=aspect, interpolation=interpolation
        )

    @staticmethod
    def _resize_fixed_size(image, height_width, interpolation=cv2.INTER_LINEAR):
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise ValueError("Wrong image supplied")
        if len(height_width) != 2:
            raise ValueError("Wrong desired size supplied")

        return cv2.resize(
            image, height_width[::-1], interpolation=interpolation
        )

    @staticmethod
    def make_coord(batch, height, width, dtype="float32"):
        height_ = 1.0 / np.array(height, "float32")
        width_ = 1.0 / np.array(width, "float32")

        vertical = (
            height_ - 1.0 + 2 * height_ * np.arange(height, dtype="float32")
        )
        horizontal = (
            width_ - 1.0 + 2 * width_ * np.arange(width, dtype="float32")
        )

        mesh = np.meshgrid(vertical, horizontal, indexing="ij")
        join = np.stack(mesh, axis=-1).astype(dtype)
        outputs = np.tile(join[None], [batch, 1, 1, 1])

        return outputs

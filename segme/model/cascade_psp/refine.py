import cv2
import numpy as np
import tensorflow as tf
from tensorflow_hub import KerasLayer


class Refiner:
    def __init__(self, hub_uri, max_size=900):
        self.model = KerasLayer(hub_uri)
        self.max_size = max_size

        self.image = tf.Variable(
            shape=(1, None, None, 3), dtype='uint8', initial_value=np.zeros((1, 0, 0, 3)).astype(np.uint8))
        self.mask = tf.Variable(
            shape=(1, None, None, 1), dtype='uint8', initial_value=np.zeros((1, 0, 0, 1)).astype(np.uint8))
        self.prev = tf.Variable(
            shape=(1, None, None, 1), dtype='uint8', initial_value=np.zeros((1, 0, 0, 1)).astype(np.uint8))

    def __call__(self, image, mask, fast=False):
        fine, coarse = self._global_step(image, mask)
        if fast:
            return fine

        return self._local_step(image, fine, coarse)

    def _global_step(self, image, mask):
        height_width = image.shape[:2]

        if max(height_width) < self.max_size:
            image = Refiner._resize_max_side(image, self.max_size, cv2.INTER_CUBIC)
            mask = Refiner._resize_max_side(mask, self.max_size, cv2.INTER_LINEAR)
        elif max(height_width) > self.max_size:
            image = Refiner._resize_max_side(image, self.max_size, cv2.INTER_AREA)
            mask = Refiner._resize_max_side(mask, self.max_size, cv2.INTER_AREA)

        fine, coarse = self._safe_predict(image, mask)

        if max(height_width) < self.max_size:
            fine = Refiner._resize_fixed_size(fine, height_width, interpolation=cv2.INTER_AREA)
            coarse = Refiner._resize_fixed_size(coarse, height_width, interpolation=cv2.INTER_AREA)
        elif max(height_width) > self.max_size:
            fine = Refiner._resize_fixed_size(fine, height_width, interpolation=cv2.INTER_LINEAR)
            coarse = Refiner._resize_fixed_size(coarse, height_width, interpolation=cv2.INTER_LINEAR)

        return fine, coarse

    def _local_step(self, image, fine, coarse, padding=16):
        height, width = fine.shape[:2]
        grid_mask = np.zeros_like(fine, dtype=np.uint32)
        grid_weight = np.zeros_like(fine, dtype=np.uint32)

        step_size = self.max_size // 2 - padding * 2
        used_start_idx = set()
        for x_idx in range(width // step_size + 1):
            for y_idx in range(height // step_size + 1):
                start_x = x_idx * step_size
                start_y = y_idx * step_size
                end_x = start_x + self.max_size
                end_y = start_y + self.max_size

                # Shift when required
                if end_x > width:
                    end_x = width
                    start_x = width - self.max_size
                if end_y > height:
                    end_y = height
                    start_y = height - self.max_size

                # Bound x/y range
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                end_x = min(width, end_x)
                end_y = min(height, end_y)

                # The same crop might appear twice due to bounding/shifting
                start_idx = start_y * width + start_x
                if start_idx in used_start_idx:
                    continue
                used_start_idx.add(start_idx)

                # Take crop
                part_image = image[start_y:end_y, start_x:end_x, :]
                part_mask = fine[start_y:end_y, start_x:end_x]
                part_prev = coarse[start_y:end_y, start_x:end_x]

                # Skip when it is not an interesting crop anyway
                part_mean = (part_mask > 127).astype(np.float32).mean()
                if part_mean > 0.9 or part_mean < 0.1:
                    continue
                grid_fine, _ = self._safe_predict(part_image, part_mask, part_prev)

                # Padding
                pred_sx = pred_sy = 0
                pred_ex = self.max_size
                pred_ey = self.max_size

                if start_x != 0:
                    start_x += padding
                    pred_sx += padding
                if start_y != 0:
                    start_y += padding
                    pred_sy += padding
                if end_x != width:
                    end_x -= padding
                    pred_ex -= padding
                if end_y != height:
                    end_y -= padding
                    pred_ey -= padding

                grid_mask[start_y:end_y, start_x:end_x] += grid_fine[pred_sy:pred_ey, pred_sx:pred_ex]
                grid_weight[start_y:end_y, start_x:end_x] += 1

        # Final full resolution output
        grid_weight_ = grid_weight.astype(np.float32) + tf.keras.backend.epsilon()
        grid_mask = np.round(grid_mask.astype(np.float32) / grid_weight_).astype(np.uint8)
        fine = np.where(grid_weight == 0, fine, grid_mask)

        return fine

    def _safe_predict(self, image, mask, prev=None):
        if len(image.shape) != 3:
            raise ValueError('Wrong image supplied')
        if len(mask.shape) != 2:
            raise ValueError('Wrong mask supplied')
        if prev is not None and len(prev.shape) != 2:
            raise ValueError('Wrong prev supplied')

        height, width = image.shape[:2]

        _image = np.pad(image, ((0, height % 8), (0, width % 8), (0, 0)))
        _mask = np.pad(mask, ((0, height % 8), (0, width % 8)))
        _prev = _mask if prev is None else np.pad(prev, ((0, height % 8), (0, width % 8)))

        self.image.assign(_image[None, ...])
        self.mask.assign(_mask[None, ..., None])
        self.prev.assign(_prev[None, ..., None])

        fine, coarse = self.model([self.image, self.mask, self.prev])
        fine, coarse = fine[0, :height, :width, 0], coarse[0, :height, :width, 0]
        fine = np.round(fine * 255).astype(np.uint8)
        coarse = np.round(coarse * 255).astype(np.uint8)

        return fine, coarse
    
    @staticmethod
    def _resize_max_side(image, max_size, interpolation=cv2.INTER_LINEAR):
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise ValueError('Wrong image supplied')

        aspect = max_size / max(image.shape[:2])

        return cv2.resize(image, (0, 0), fx=aspect, fy=aspect, interpolation=interpolation)
    
    @staticmethod
    def _resize_fixed_size(image, height_width, interpolation=cv2.INTER_LINEAR):
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise ValueError('Wrong image supplied')
        if len(height_width) != 2:
            raise ValueError('Wrong desired size supplied')

        return cv2.resize(image, height_width[::-1], interpolation=interpolation)

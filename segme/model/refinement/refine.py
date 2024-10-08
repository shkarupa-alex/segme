import logging
import math
from functools import partial
from itertools import starmap

import cv2
import numpy as np


class BatchedRefiner:
    def __init__(
        self,
        global_stage,
        scale_stage,
        local_stage,
        batch_size,
        pretrain_size,
        scale_step=2,
        padding_size=32,
    ):
        self.global_stage = global_stage
        self.scale_stage = scale_stage
        self.local_stage = local_stage
        self.batch_size = batch_size
        self.pretrain_size = pretrain_size
        self.scale_step = scale_step
        self.padding_size = padding_size

        if not self.global_stage and not self.local_stage:
            raise ValueError(
                "At least one of global/local refinement mode "
                "should be active."
            )

        self.queue = []

    def __call__(self, image, mask, key=None):
        if 3 != len(image.shape) or 3 != image.shape[-1]:
            raise ValueError("Wrong image supplied")
        if "uint8" != image.dtype:
            raise ValueError("Wrong image dtype")

        if 3 == len(mask.shape) and 1 == mask.shape[-1]:
            mask = np.squeeze(mask, axis=-1)
        elif 2 != len(mask.shape):
            raise ValueError("Wrong mask supplied")
        if mask.dtype != "uint8":
            raise ValueError("Wrong mask dtype")

        self.queue.append((image, mask, key))

        return self._check_queue(False)

    def __del__(self):
        if self.queue:
            logging.warning(
                "Queue is not empty. "
                "Use `.tail()` to read the rest of refined masks.",
                exc_info=True,
            )

    def tail(self):
        return self._check_queue(True)

    def _check_queue(self, tail):
        if not self.queue:
            return []

        if not tail and len(self.queue) < self.batch_size:
            return []

        images, masks, keys = zip(*self.queue)
        self.queue = []

        masks = self._global_refine(images, masks)
        masks = self._scale_refine(images, masks)
        masks = self._local_refine(images, masks)

        return list(zip(masks, keys))

    def _global_refine(self, images, masks):
        if not self.global_stage:
            return masks

        resize_to_max = partial(
            self._resize_to_max, max_size=self.pretrain_size, pad_square=True
        )
        images_, masks_, sizes_, paddings_ = zip(
            *starmap(resize_to_max, zip(images, masks))
        )
        images_ = np.stack(images_, axis=0)
        masks_ = np.stack(masks_, axis=0)[..., None]

        fines_ = self._refine_batch(images_, masks_, False)

        fines_ = np.squeeze(fines_, axis=-1)
        fines = starmap(self._resize_to_src, zip(fines_, sizes_, paddings_))

        return fines

    def _scale_refine(self, images, masks):
        if not self.scale_stage:
            return masks

        sizes = {}
        for i, image in enumerate(images):
            max_size = max(image.shape[:2])
            max_scale = math.log(max_size / self.pretrain_size, self.scale_step)
            for scale in range(1, round(max_scale)):
                size = self.pretrain_size * self.scale_step**scale
                sizes[size] = sizes.get(size, [])
                sizes[size].append(i)

        fines = list(masks)
        for size in sorted(sizes.keys()):
            images_ = [images[i] for i in sizes[size]]
            fines_ = [fines[i] for i in sizes[size]]

            resize_to_max = partial(
                self._resize_to_max, max_size=size, pad_square=False
            )
            images_, fines_, sizes_, padding_ = zip(
                *starmap(resize_to_max, zip(images_, fines_))
            )

            fines_ = self._local_refine(images_, fines_)

            fines_ = starmap(self._resize_to_src, zip(fines_, sizes_, padding_))
            for i, fine_ in zip(sizes[size], fines_):
                fines[i] = fine_

        return fines

    def _local_refine(self, images, masks):
        if not self.local_stage:
            return masks

        crops = list(self._make_crops(images, masks))

        fines = []
        for i in range(0, len(crops), self.batch_size):
            images_, masks_ = zip(*crops[i : i + self.batch_size])
            images_ = np.stack(images_, axis=0)
            masks_ = np.stack(masks_, axis=0)[..., None]
            fines_ = self._refine_batch(images_, masks_, True)
            fines.extend(np.squeeze(fines_, axis=-1))

        fines = self._join_crops(images, masks, fines)

        return fines

    def _resize_to_max(self, image, mask, max_size, pad_square):
        size = image.shape[:2]

        ratio = max_size / max(size)
        if ratio > 1.0:
            interpolation = cv2.INTER_CUBIC
        elif ratio < 1.0:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = None

        if interpolation is not None:
            image = cv2.resize(
                image,
                (0, 0),
                fx=ratio,
                fy=ratio,
                interpolation=interpolation,
            )
            mask = cv2.resize(
                mask,
                (0, 0),
                fx=ratio,
                fy=ratio,
                interpolation=interpolation,
            )

        if pad_square:
            padding = max_size - np.array(image.shape[:2])
            padding = np.array([padding // 2, padding - padding // 2]).T
            padding = padding.ravel().tolist()

            image = cv2.copyMakeBorder(
                image,
                *padding,
                borderType=cv2.BORDER_REFLECT,
            )
            mask = cv2.copyMakeBorder(
                mask,
                *padding,
                borderType=cv2.BORDER_REFLECT,
            )
        else:
            padding = [0] * 4

        mask = (mask > 127).astype("uint8") * 255

        return image, mask, size, padding

    def _resize_to_src(self, mask, src_size, pad_size):
        mask = mask[
            pad_size[0] : mask.shape[0] - pad_size[1],
            pad_size[2] : mask.shape[1] - pad_size[3],
        ]

        max_size = max(mask.shape[:2])
        if max(src_size) < max_size:
            interpolation = cv2.INTER_AREA
        elif max(src_size) > max_size:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = None

        if interpolation is not None:
            mask = cv2.resize(mask, src_size[::-1], interpolation=interpolation)

        mask = (mask > 127).astype("uint8") * 255

        return mask

    def _make_crops(self, images, masks):
        for image, mask in zip(images, masks):
            height, width = image.shape[:2]
            if min(height, width) < self.pretrain_size:
                continue

            used_idxs = set()
            step_size = self.pretrain_size // 2 - self.padding_size * 2
            for y_idx in range(height // step_size + 1):
                for x_idx in range(width // step_size + 1):
                    start_y = y_idx * step_size
                    start_x = x_idx * step_size
                    end_y = start_y + self.pretrain_size
                    end_x = start_x + self.pretrain_size

                    # Shift when required
                    if end_y > height:
                        end_y = height
                        start_y = end_y - self.pretrain_size
                    if end_x > width:
                        end_x = width
                        start_x = end_x - self.pretrain_size

                    # The same crop might appear twice due to bounding/shifting
                    idx_key = start_y * (width // step_size + 1) + start_x
                    if idx_key in used_idxs:
                        continue
                    used_idxs.add(idx_key)

                    # Take crop
                    image_crop = image[start_y:end_y, start_x:end_x]
                    mask_crop = mask[start_y:end_y, start_x:end_x]
                    if 1 == len(np.unique(mask_crop)):
                        continue

                    yield image_crop, mask_crop

    def _join_crops(self, images, masks, crops):
        for image, mask in zip(images, masks):
            height, width = image.shape[:2]
            if min(height, width) < self.pretrain_size:
                yield mask

            fine = np.zeros_like(mask, dtype=np.uint32)
            weight = np.zeros_like(mask, dtype=np.uint32)

            used_idxs = set()
            step_size = self.pretrain_size // 2 - self.padding_size * 2
            for y_idx in range(height // step_size + 1):
                for x_idx in range(width // step_size + 1):
                    start_y = y_idx * step_size
                    start_x = x_idx * step_size
                    end_y = start_y + self.pretrain_size
                    end_x = start_x + self.pretrain_size

                    # Shift when required
                    if end_y > height:
                        end_y = height
                        start_y = end_y - self.pretrain_size
                    if end_x > width:
                        end_x = width
                        start_x = end_x - self.pretrain_size

                    # The same crop might appear twice due to bounding/shifting
                    idx_key = start_y * (width // step_size + 1) + start_x
                    if idx_key in used_idxs:
                        continue
                    used_idxs.add(idx_key)

                    # Take crop
                    mask_crop = mask[start_y:end_y, start_x:end_x]
                    if 1 == len(np.unique(mask_crop)):
                        continue

                    fine_crop = crops.pop(0)

                    # Padding
                    pred_sy = pred_sx = 0
                    pred_ey = pred_ex = self.pretrain_size

                    if start_y != 0:
                        start_y += self.padding_size
                        pred_sy += self.padding_size
                    if start_x != 0:
                        start_x += self.padding_size
                        pred_sx += self.padding_size
                    if end_y != height:
                        end_y -= self.padding_size
                        pred_ey -= self.padding_size
                    if end_x != width:
                        end_x -= self.padding_size
                        pred_ex -= self.padding_size

                    fine[start_y:end_y, start_x:end_x] += fine_crop[
                        pred_sy:pred_ey, pred_sx:pred_ex
                    ]
                    weight[start_y:end_y, start_x:end_x] += 1

            # Final full resolution output
            judge = np.where(weight % 2 == 0)
            fine[judge] += mask[judge]
            weight[judge] += 1

            fine = fine.astype("float32") / (weight.astype("float32") + 1e-7)
            fine = (np.round(fine) > 127).astype("uint8") * 255
            fine = np.where(weight == 0, mask, fine)

            yield fine

    def _refine_batch(self, images, masks, crop_mode):
        raise NotImplementedError

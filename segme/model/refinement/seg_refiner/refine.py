import numpy as np
from keras.src import backend

from segme.backend import model_inference_fn
from segme.model.refinement.refine import BatchedRefiner
from segme.model.refinement.seg_refiner.model import SegRefiner


class Refiner(BatchedRefiner):
    def __init__(
        self,
        weights_global="seg_refiner__lr__256",
        weights_local="seg_refiner__hr__256",
        batch_size=8,
        pretrain_size=256,
    ):
        super().__init__(
            weights_global is not None,
            weights_local is not None,
            weights_local is not None,
            batch_size,
            pretrain_size,
        )

        self.model_global = model_inference_fn(
            SegRefiner(weights=weights_global), jit_compile=True
        )
        if weights_global == weights_local:
            self.model_local = self.model_global
        else:
            self.model_local = model_inference_fn(
                SegRefiner(weights=weights_local), jit_compile=True
            )

        self.num_steps = 6
        self.betas_cumprod = np.linspace(0.8, 0.0, self.num_steps)
        self.betas_cumprod_prev = np.insert(self.betas_cumprod[:-1], 0, 1.0)

    def _refine_batch(self, images, masks, crop_mode):
        return self._p_sample_loop(images, masks, crop_mode)

    def _p_sample_loop(self, images, masks, crop_mode):
        curr_probs = np.zeros_like(masks, dtype="float32")
        sample_noise = np.random.uniform(
            size=(self.num_steps - 1,) + masks.shape
        )

        fines = masks
        for time in range(self.num_steps)[::-1]:
            fines, curr_probs = self._p_sample(
                images, fines, time, curr_probs, crop_mode
            )

            if time > 0:
                sample_map = sample_noise[time - 1] < curr_probs
                sample_map = sample_map.astype("uint8")
                fines = fines * sample_map + masks * (1 - sample_map)

        return fines

    def _p_sample(self, images, masks, time, curr_probs, crop_mode):
        beta_cumprod = self.betas_cumprod[time]
        beta_cumprod_prev = self.betas_cumprod_prev[time]

        fines = self._predict_step(images, masks, time, crop_mode)

        x_start_probs = np.abs(fines - 0.5) * 2.0
        p_c_to_f = (
            x_start_probs
            * (beta_cumprod_prev - beta_cumprod)
            / (1.0 - x_start_probs * beta_cumprod)
        )
        curr_probs = curr_probs + (1.0 - curr_probs) * p_c_to_f

        fines = np.round((fines >= 0.5) * 255.0).astype("uint8")

        return fines, curr_probs

    def _predict_step(self, images, masks, time, crop_mode):
        current_model = self.model_local if crop_mode else self.model_global
        fine = current_model(
            backend.convert_to_tensor(images),
            backend.convert_to_tensor(masks),
            backend.convert_to_tensor([time] * images.shape[0], "int32"),
        )
        if isinstance(fine, (list, tuple)):
            if 1 != len(fine):
                raise ValueError("Unexpected model inference output.")
            fine = fine[0]

        fine = backend.convert_to_numpy(fine)

        return fine

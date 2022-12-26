import albumentations as alb
import cv2
import json
import numpy as np


class RotateFix(alb.Rotate):
    def __init__(self, limit=90, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None,
                 mask_value=None, rotate_method='largest_box', crop_border=False, always_apply=False, p=0.5):
        super().__init__(limit=limit, interpolation=interpolation, border_mode=border_mode,
                         value=value, mask_value=mask_value, rotate_method=rotate_method, crop_border=crop_border,
                         always_apply=always_apply, p=p)

    def apply_to_mask(self, img, angle=0, x_min=None, x_max=None, y_min=None, y_max=None, **params):
        img_out = alb.rotate(img, angle, cv2.INTER_NEAREST_EXACT, self.border_mode, self.mask_value)
        if self.crop_border:
            img_out = alb.crop(img_out, x_min, y_min, x_max, y_max)
        return img_out


class ElasticTransformFix(alb.ElasticTransform):
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, value=None, mask_value=None, always_apply=False, approximate=False,
                 same_dxdy=False, p=0.5):
        super().__init__(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, interpolation=interpolation,
                         border_mode=border_mode, value=value, mask_value=mask_value, always_apply=always_apply,
                         approximate=approximate, same_dxdy=same_dxdy, p=p)

    def apply_to_mask(self, img, random_state=None, **params):
        return alb.elastic_transform(
            img, self.alpha, self.sigma, self.alpha_affine, cv2.INTER_NEAREST_EXACT, self.border_mode, self.mask_value,
            np.random.RandomState(random_state), self.approximate, self.same_dxdy)


class GridDistortionFix(alb.ElasticTransform):
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, value=None, mask_value=None, always_apply=False, approximate=False,
                 same_dxdy=False, p=0.5):
        super().__init__(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, interpolation=interpolation,
                         border_mode=border_mode, value=value, mask_value=mask_value, always_apply=always_apply,
                         approximate=approximate, same_dxdy=same_dxdy, p=p)

    def apply_to_mask(self, img, random_state=None, **params):
        return alb.elastic_transform(
            img, self.alpha, self.sigma, self.alpha_affine, cv2.INTER_NEAREST_EXACT, self.border_mode, self.mask_value,
            np.random.RandomState(random_state), self.approximate, self.same_dxdy)


def drop_unapplied(replay, as_json=True):
    if not replay['applied']:
        return []

    if 'transforms' in replay:
        transforms = []
        for r in replay['transforms']:
            t = drop_unapplied(r, as_json=False)
            transforms.extend(t)
        skip = {'ReplayCompose', 'OneOf', 'Resize', 'PadIfNeeded', 'RandomCrop'}
        transforms = [t for t in transforms if t['__class_fullname__'].split('.')[-1] not in skip]
        return transforms

    del replay['applied']
    del replay['always_apply']

    if as_json:
        return json.dumps([replay], indent=2, sort_keys=True, cls=NumpyEncoder)

    return [replay]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

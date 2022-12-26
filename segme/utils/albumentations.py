import json
import numpy as np


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

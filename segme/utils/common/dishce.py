import cv2
import numpy as np
from skimage.measure import label
from skimage.morphology import disk
from skimage.morphology import skeletonize


def _filter_boundaries(contours, mask, condition):
    condition = cv2.dilate(condition.astype("uint8"), disk(1)).astype("bool")
    components = label(mask)  # Connected regions
    labels = np.unique(components)  # Indices of the connected regions
    independent = np.ones(labels.shape[0])  # Label of each connected regions
    independent[0] = 0  # Indicates the background region

    boundaries = []
    independent_map = np.zeros(condition.shape[0:2])

    for i in range(len(contours)):
        boundaries_, boundary_ = [], []

        for j in range(contours[i].shape[0]):
            r, c = contours[i][j, 0, 1], contours[i][j, 0, 0]

            if not condition[r, c].sum() or independent_map[r, c]:
                if len(boundary_) > 0:
                    boundaries_.append(boundary_)
                    boundary_ = []
                continue

            boundary_.append([c, r])
            independent_map[r, c] += 1
            independent[components[r, c]] = (
                0  # Part of the boundary of this region needs human correction
            )

        if len(boundary_) > 0:
            boundaries_.append(boundary_)

        # Check if the first and the last boundaries are connected. If yes,
        # invert the first boundary and attach it after the last boundary
        if len(boundaries_) > 1:
            first_x, first_y = boundaries_[0][0]
            last_x, last_y = boundaries_[-1][-1]
            if (
                abs(first_x - last_x) == 1
                and first_y == last_y
                or first_x == last_x
                and abs(first_y - last_y) == 1
                or abs(first_x - last_x) == 1
                and abs(first_y - last_y) == 1
            ):
                boundaries_[-1].extend(boundaries_[0][::-1])
                del boundaries_[0]

        for k in range(len(boundaries_)):
            boundaries_[k] = np.array(boundaries_[k])[:, None, :]

        boundaries.extend(boundaries_)

    return boundaries, np.sum(independent)


def _approximate_rdp(boundaries, epsilon=1.0):
    # Approximate each boundary by DP algorithm
    # https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm

    total_points = 0

    # Polygon approximate of each boundary and count the total control
    # points number of all the boundaries
    for i in range(len(boundaries)):
        total_points += len(cv2.approxPolyDP(boundaries[i], epsilon, False))

    return total_points


def _relax_hce(y_true, y_pred, y_true_skeleton, relax=5, epsilon=2.0):
    # Compute statistics
    union = y_true | y_pred
    tp = y_true & y_pred
    fp = y_pred ^ tp
    fn = y_true ^ tp

    # Relax the union of y_true and y_pred
    union_ = cv2.erode(union.astype("uint8"), disk(1), iterations=relax).astype(
        "bool"
    )

    # Get the relaxed False Positive regions for computing the human efforts
    # in correcting them
    fp_ = fp & union_
    for i in range(relax):
        fp_ = cv2.dilate(fp_.astype("uint8"), disk(1)).astype("bool")
        fp_ &= ~(tp | fn)
    fp_ &= fp

    # Get the relaxed False Negative regions for computing the human efforts
    # in correcting them
    fn_ = fn & union_
    for i in range(relax):
        fn_ = cv2.dilate(fn_.astype("uint8"), disk(1)).astype("bool")
        fn_ &= ~(tp | fp)
    fn_ &= fn

    # Preserve the structural components of FN
    fn_ |= y_true_skeleton ^ (tp & y_true_skeleton)

    # Find independent region points
    contours_fp, _ = cv2.findContours(
        fp_.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    boundaries_fp, indep_fp_cnt = _filter_boundaries(contours_fp, fp_, tp | fn_)

    contours_fn, _ = cv2.findContours(
        fn_.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    boundaries_fn, indep_fn_cnt = _filter_boundaries(
        contours_fn, fn_, ~(tp | fp_ | fn_)
    )

    # Find exact polygon control points
    poly_fp_cnt = _approximate_rdp(boundaries_fp, epsilon=epsilon)
    poly_fn_cnt = _approximate_rdp(boundaries_fn, epsilon=epsilon)

    return poly_fp_cnt, indep_fp_cnt, poly_fn_cnt, indep_fn_cnt


def compute_hce(y_true, y_pred, y_true_skeleton=None):
    # Binarize y_true
    if 3 == len(y_true.shape) and 1 == y_true.shape[-1]:
        y_true = y_true[:, :, 0]
    assert 2 == len(y_true.shape)
    y_true = y_true > 127

    # Binarize y_pred
    if 3 == len(y_pred.shape) and 1 == y_pred.shape[-1]:
        y_pred = y_pred[:, :, 0]
    assert 2 == len(y_pred.shape)
    y_pred = y_pred > 127

    # Create y_true_skeleton
    if y_true_skeleton is None:
        y_true_skeleton = skeletonize(y_true)
    else:
        y_true_skeleton = y_true_skeleton.astype("bool")

    # Binarize y_true_skeleton
    if 3 == len(y_true_skeleton.shape) and 1 == y_true_skeleton.shape[-1]:
        y_true_skeleton = y_true_skeleton[:, :, 0]
    assert 2 == len(y_true_skeleton.shape)

    points = _relax_hce(y_true, y_pred, y_true_skeleton)
    hce = sum(points)

    return hce

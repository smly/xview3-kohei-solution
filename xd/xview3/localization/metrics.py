import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, count=1):
        self.sum += val
        self.count += count
        self.avg = self.sum / self.count


def get_boxinfo_from_binar_map(out_bin, min_area=3):
    binar_numpy = out_bin.squeeze().astype(np.uint8)
    assert binar_numpy.ndim == 2
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binar_numpy, connectivity=4
    )

    boxes = stats[1:, :]
    points = centroids[1:, :]
    index = boxes[:, 4] >= min_area
    boxes = boxes[index]
    points = points[index]

    return {
        "num": len(points),
        "points": points.tolist(),
        "boxes": boxes.tolist(),
    }


def calculate_p_r_f(tp, fp, fn):
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0

    try:
        fscore = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        fscore = 0
    if precision == np.nan or recall == np.nan or fscore == np.nan:
        return 0, 0, 0
    else:
        return precision, recall, fscore


def evaluate_metrics(pred_points, gt_points):
    pred_array = np.array(pred_points)
    gt_array = np.array(gt_points)

    if pred_array.shape[0] == 0 and gt_array.shape[0] > 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": gt_array.shape[0],
            "recall": 0,
            "precision": 0,
            "fscore": 0,
        }
    elif pred_array.shape[0] > 0 and gt_array.shape[0] == 0:
        return {
            "tp": 0,
            "fp": pred_array.shape[0],
            "fn": 0,
            "recall": 0,
            "precision": 0,
            "fscore": 0,
        }
    elif pred_array.shape[0] == 0 and gt_array.shape[0] == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "recall": 0,
            "precision": 0,
            "fscore": 0,
        }
    else:
        # predIdx x gtIdx
        dist_mat = distance_matrix(pred_array, gt_array, p=2) * 10
        # 線形割当問題
        rows, cols = linear_sum_assignment(dist_mat)
        # pred_array, gt_array の index
        tp_inds = [
            {"pred_idx": rows[ii], "gt_idx": cols[ii]}
            for ii in range(len(rows))
            if dist_mat[rows[ii], cols[ii]] < 200
        ]
        tp_pred_inds = [a["pred_idx"] for a in tp_inds]
        tp_gt_inds = [a["gt_idx"] for a in tp_inds]

        # pred_array の index
        fp_inds = [
            a for a in range(pred_array.shape[0]) if a not in tp_pred_inds
        ]  # noqa
        # gt_array の index
        fn_inds = [a for a in range(gt_array.shape[0]) if a not in tp_gt_inds]

        tp, fp, fn = len(tp_inds), len(fp_inds), len(fn_inds)
        prec, recall, fscore = calculate_p_r_f(tp, fp, fn)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": recall,
            "precision": prec,
            "fscore": fscore,
        }

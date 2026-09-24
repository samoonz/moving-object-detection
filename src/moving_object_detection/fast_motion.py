from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class FastConfig:
    feature_rows: int = 6
    feature_cols: int = 10
    features_per_cell: int = 30
    feature_quality: float = 0.01
    feature_min_distance: int = 8
    lk_win_size: int = 21
    lk_max_level: int = 3
    fb_max_error: float = 1.5
    max_planes: int = 2
    min_plane_points: int = 40
    secondary_min_coverage: float = 0.25
    homography_threshold_px: float = 1.5
    magsac_confidence: float = 0.999
    residual_floor: float = 10.0
    residual_mad_k: float = 4.0
    gradient_weight: float = 0.25
    valid_erode: int = 7
    blur_ksize: int = 5


def _odd(x: int) -> int:
    x = max(1, int(x))
    return x if x % 2 else x + 1


def _grid_features(gray: np.ndarray, cfg: FastConfig) -> np.ndarray | None:
    h, w = gray.shape
    out = []
    for r in range(cfg.feature_rows):
        y0 = int(r * h / cfg.feature_rows)
        y1 = int((r + 1) * h / cfg.feature_rows)
        for c in range(cfg.feature_cols):
            x0 = int(c * w / cfg.feature_cols)
            x1 = int((c + 1) * w / cfg.feature_cols)
            roi = gray[y0:y1, x0:x1]
            pts = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=cfg.features_per_cell,
                qualityLevel=cfg.feature_quality,
                minDistance=cfg.feature_min_distance,
                blockSize=7,
            )
            if pts is None:
                continue
            pts[:, 0, 0] += x0
            pts[:, 0, 1] += y0
            out.append(pts)
    if not out:
        return None
    return np.concatenate(out, axis=0).astype(np.float32)


def _track(prev: np.ndarray, cur: np.ndarray, cfg: FastConfig) -> Tuple[np.ndarray, np.ndarray]:
    p0 = _grid_features(prev, cfg)
    if p0 is None or len(p0) < 8:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    lk = dict(
        winSize=(_odd(cfg.lk_win_size), _odd(cfg.lk_win_size)),
        maxLevel=cfg.lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
    )
    p1, s1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **lk)
    if p1 is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    p0b, s2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **lk)
    if p0b is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    a = p0.reshape(-1, 2)
    b = p1.reshape(-1, 2)
    back = p0b.reshape(-1, 2)
    fb = np.linalg.norm(a - back, axis=1)
    good = (
        (s1.reshape(-1) == 1)
        & (s2.reshape(-1) == 1)
        & np.isfinite(b).all(axis=1)
        & (fb <= cfg.fb_max_error)
    )
    return a[good], b[good]


def _coverage(points: np.ndarray, w: int, h: int) -> float:
    if len(points) < 2:
        return 0.0
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return float(max(x1 - x0, 0) * max(y1 - y0, 0) / max(w * h, 1))


def _fit_planes(p0: np.ndarray, p1: np.ndarray, shape: Tuple[int, int], cfg: FastConfig):
    h, w = shape
    if len(p0) < cfg.min_plane_points:
        return [], []

    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    remain0 = p0.copy()
    remain1 = p1.copy()
    planes: List[np.ndarray] = []
    infos: List[Dict[str, float | int]] = []

    for plane_idx in range(cfg.max_planes):
        if len(remain0) < cfg.min_plane_points:
            break
        try:
            H, mask = cv2.findHomography(
                remain0,
                remain1,
                method,
                cfg.homography_threshold_px,
                maxIters=5000,
                confidence=cfg.magsac_confidence,
            )
        except TypeError:
            H, mask = cv2.findHomography(
                remain0, remain1, method, cfg.homography_threshold_px
            )
        if H is None or mask is None or not np.isfinite(H).all():
            break

        inliers = mask.reshape(-1).astype(bool)
        n_in = int(inliers.sum())
        if n_in < cfg.min_plane_points:
            break

        cov = _coverage(remain0[inliers], w, h)
        if plane_idx > 0 and cov < cfg.secondary_min_coverage:
            break

        H = H.astype(np.float64)
        H /= max(H[2, 2], 1e-12)
        planes.append(H)
        infos.append(
            {
                "inliers": n_in,
                "inlier_ratio": float(n_in / max(len(remain0), 1)),
                "coverage": cov,
            }
        )

        remain0 = remain0[~inliers]
        remain1 = remain1[~inliers]

    return planes, infos


def _robust_threshold(values: np.ndarray, floor: float, k: float) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float(floor)
    # Lower 70% is treated as background to stop moving foreground from
    # inflating the threshold when the object becomes large.
    cutoff = np.quantile(v, 0.70)
    bg = v[v <= cutoff]
    med = float(np.median(bg))
    mad = float(np.median(np.abs(bg - med)))
    return max(float(floor), med + k * 1.4826 * mad)


class FastMotionEstimator:
    """
    Sparse-camera-motion compensation + multi-homography residual detector.

    It is intentionally much cheaper than dense optical flow:
      sparse features -> LK tracks -> 1-2 background homographies -> warped
      frame residual. Multiple wide-coverage planes reduce false positives
      from parallax without fitting a dense flow field.
    """

    def __init__(self, cfg: FastConfig):
        self.cfg = cfg
        cv2.setUseOptimized(True)

    def estimate(self, prev_bgr: np.ndarray, cur_bgr: np.ndarray):
        cfg = self.cfg
        k = _odd(cfg.blur_ksize)
        prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        cur = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
        if k > 1:
            prev = cv2.GaussianBlur(prev, (k, k), 0)
            cur = cv2.GaussianBlur(cur, (k, k), 0)

        p0, p1 = _track(prev, cur, cfg)
        planes, plane_info = _fit_planes(p0, p1, prev.shape, cfg)

        if not planes:
            A, mask = (None, None)
            if len(p0) >= 8:
                A, mask = cv2.estimateAffinePartial2D(
                    p0,
                    p1,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=cfg.homography_threshold_px,
                    maxIters=3000,
                    confidence=cfg.magsac_confidence,
                    refineIters=10,
                )
            if A is not None:
                H = np.vstack([A, [0.0, 0.0, 1.0]])
                planes = [H]
                plane_info = [{
                    "inliers": int(mask.sum()) if mask is not None else 0,
                    "inlier_ratio": float(mask.mean()) if mask is not None else 0.0,
                    "coverage": 0.0,
                }]
            else:
                planes = [np.eye(3, dtype=np.float64)]
                plane_info = [{"inliers": 0, "inlier_ratio": 0.0, "coverage": 0.0}]

        h, w = prev.shape
        source_valid = np.full((h, w), 255, np.uint8)

        residuals = []
        valids = []
        cur_f = cur.astype(np.float32)
        gx_cur = cv2.Sobel(cur_f, cv2.CV_32F, 1, 0, ksize=3)
        gy_cur = cv2.Sobel(cur_f, cv2.CV_32F, 0, 1, ksize=3)

        for H in planes:
            warped = cv2.warpPerspective(
                prev,
                H,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            valid = cv2.warpPerspective(
                source_valid,
                H,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
            if cfg.valid_erode > 1:
                valid = cv2.erode(
                    valid,
                    np.ones((_odd(cfg.valid_erode), _odd(cfg.valid_erode)), np.uint8),
                    iterations=1,
                )

            valid_bool = valid > 0
            warped_f = warped.astype(np.float32)

            if np.any(valid_bool):
                exposure_shift = float(np.median(cur_f[valid_bool] - warped_f[valid_bool]))
                exposure_shift = float(np.clip(exposure_shift, -30.0, 30.0))
            else:
                exposure_shift = 0.0

            warped_f = np.clip(warped_f + exposure_shift, 0.0, 255.0)
            photo = np.abs(cur_f - warped_f)

            gx_prev = cv2.Sobel(warped_f, cv2.CV_32F, 1, 0, ksize=3)
            gy_prev = cv2.Sobel(warped_f, cv2.CV_32F, 0, 1, ksize=3)
            grad = 0.5 * (np.abs(gx_cur - gx_prev) + np.abs(gy_cur - gy_prev))

            residual = photo + cfg.gradient_weight * grad
            residual[~valid_bool] = np.inf
            residuals.append(residual)
            valids.append(valid_bool)

        residual_stack = np.stack(residuals, axis=0)
        residual = np.min(residual_stack, axis=0)
        valid = np.any(np.stack(valids, axis=0), axis=0)
        residual[~valid] = np.nan

        threshold = _robust_threshold(
            residual[valid],
            cfg.residual_floor,
            cfg.residual_mad_k,
        )
        score = residual / max(threshold, 1e-6)
        score[~np.isfinite(score)] = 0.0

        meta = {
            "geometry_model": "fast-multi-homography",
            "tracked_features": int(len(p0)),
            "planes": int(len(planes)),
            "plane_info": plane_info,
            "residual_threshold_px": float(threshold),
        }
        return score.astype(np.float32), valid, meta

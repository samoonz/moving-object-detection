from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class GeometryConfig:
    sample_stride: int = 8
    magsac_confidence: float = 0.999
    magsac_max_iters: int = 10000
    homography_threshold_px: float = 1.5
    fundamental_threshold_px: float = 0.9
    min_model_inlier_ratio: float = 0.35
    homography_strong_inlier_ratio: float = 0.85
    prefer_fundamental_margin: float = 0.05
    min_residual_px: float = 1.0
    adaptive_mad_k: float = 4.0
    max_flow_px: float = 1000.0
    fb_consistency_max_px: float = 8.0


def _grid_correspondences(flow: np.ndarray, stride: int, max_flow: float):
    h, w = flow.shape[:2]
    ys = np.arange(stride // 2, h, stride, dtype=np.float32)
    xs = np.arange(stride // 2, w, stride, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    p1 = np.stack([xx.ravel(), yy.ravel()], axis=1)
    sampled = flow[yy.astype(np.int32), xx.astype(np.int32)].reshape(-1, 2)
    p2 = p1 + sampled
    mag = np.linalg.norm(sampled, axis=1)
    valid = np.isfinite(p2).all(axis=1) & (mag < max_flow)
    return p1[valid], p2[valid]


def _fit_h(p1: np.ndarray, p2: np.ndarray, cfg: GeometryConfig):
    if len(p1) < 8:
        return None, 0.0
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    try:
        H, m = cv2.findHomography(
            p1, p2, method, cfg.homography_threshold_px,
            maxIters=cfg.magsac_max_iters, confidence=cfg.magsac_confidence,
        )
    except TypeError:
        H, m = cv2.findHomography(p1, p2, method, cfg.homography_threshold_px)
    ratio = float(m.mean()) if m is not None else 0.0
    return H, ratio


def _fit_f(p1: np.ndarray, p2: np.ndarray, cfg: GeometryConfig):
    if len(p1) < 12:
        return None, 0.0
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.FM_RANSAC
    try:
        Fm, m = cv2.findFundamentalMat(
            p1, p2, method,
            cfg.fundamental_threshold_px,
            cfg.magsac_confidence,
            cfg.magsac_max_iters,
        )
    except TypeError:
        Fm, m = cv2.findFundamentalMat(
            p1, p2, method,
            cfg.fundamental_threshold_px,
            cfg.magsac_confidence,
        )
    if Fm is not None and Fm.shape != (3, 3):
        Fm = Fm[:3, :3]
    ratio = float(m.mean()) if m is not None else 0.0
    return Fm, ratio


def _dense_coords(flow: np.ndarray):
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x2 = xx + flow[..., 0]
    y2 = yy + flow[..., 1]
    return xx, yy, x2, y2


def _homography_error(flow: np.ndarray, H: np.ndarray) -> np.ndarray:
    xx, yy, x2, y2 = _dense_coords(flow)
    den = H[2, 0] * xx + H[2, 1] * yy + H[2, 2]
    den = np.where(np.abs(den) < 1e-6, 1e-6, den)
    px = (H[0, 0] * xx + H[0, 1] * yy + H[0, 2]) / den
    py = (H[1, 0] * xx + H[1, 1] * yy + H[1, 2]) / den
    return np.sqrt((x2 - px) ** 2 + (y2 - py) ** 2).astype(np.float32)


def _fundamental_error(flow: np.ndarray, Fm: np.ndarray) -> np.ndarray:
    xx, yy, x2, y2 = _dense_coords(flow)
    a = Fm[0, 0] * xx + Fm[0, 1] * yy + Fm[0, 2]
    b = Fm[1, 0] * xx + Fm[1, 1] * yy + Fm[1, 2]
    c = Fm[2, 0] * xx + Fm[2, 1] * yy + Fm[2, 2]
    num = np.abs(x2 * a + y2 * b + c)

    at = Fm[0, 0] * x2 + Fm[1, 0] * y2 + Fm[2, 0]
    bt = Fm[0, 1] * x2 + Fm[1, 1] * y2 + Fm[2, 1]
    den = np.sqrt(a * a + b * b + at * at + bt * bt + 1e-6)
    return (num / den).astype(np.float32)


def _robust_threshold(values: np.ndarray, floor: float, k: float) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float(floor)
    cutoff = np.quantile(v, 0.65)
    bg = v[v <= cutoff]
    med = float(np.median(bg))
    mad = float(np.median(np.abs(bg - med)))
    return max(float(floor), med + k * 1.4826 * mad)


def _fb_valid(fwd: np.ndarray, bwd: Optional[np.ndarray], max_err: float) -> np.ndarray:
    h, w = fwd.shape[:2]
    if bwd is None:
        return np.ones((h, w), dtype=bool)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = xx + fwd[..., 0]
    my = yy + fwd[..., 1]
    bwx = cv2.remap(bwd[..., 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    bwy = cv2.remap(bwd[..., 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    err = np.sqrt((fwd[..., 0] + bwx) ** 2 + (fwd[..., 1] + bwy) ** 2)
    inside = (mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1)
    return inside & np.isfinite(err) & (err <= max_err)


def residual_motion_score(
    fwd: np.ndarray,
    bwd: Optional[np.ndarray],
    cfg: GeometryConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float | str]]:
    p1, p2 = _grid_correspondences(fwd, cfg.sample_stride, cfg.max_flow_px)
    H, hr = _fit_h(p1, p2, cfg)
    Fm, fr = _fit_f(p1, p2, cfg)

    choose_f = (
        Fm is not None
        and fr >= cfg.min_model_inlier_ratio
        and (
            H is None
            or (
                hr < cfg.homography_strong_inlier_ratio
                and fr >= hr + cfg.prefer_fundamental_margin
            )
        )
    )

    if choose_f:
        model = "fundamental"
        err = _fundamental_error(fwd, Fm)
        floor = cfg.fundamental_threshold_px
    elif H is not None and hr >= cfg.min_model_inlier_ratio:
        model = "homography"
        err = _homography_error(fwd, H)
        floor = cfg.homography_threshold_px
    elif Fm is not None:
        model = "fundamental-low-support"
        err = _fundamental_error(fwd, Fm)
        floor = cfg.fundamental_threshold_px
    elif H is not None:
        model = "homography-low-support"
        err = _homography_error(fwd, H)
        floor = cfg.homography_threshold_px
    else:
        model = "median-flow-fallback"
        med = np.nanmedian(fwd.reshape(-1, 2), axis=0)
        err = np.linalg.norm(fwd - med, axis=2).astype(np.float32)
        floor = cfg.min_residual_px

    thr = _robust_threshold(
        err[:: cfg.sample_stride, :: cfg.sample_stride],
        max(cfg.min_residual_px, floor),
        cfg.adaptive_mad_k,
    )
    score = err / max(thr, 1e-6)
    valid = _fb_valid(fwd, bwd, cfg.fb_consistency_max_px)
    finite = np.isfinite(score) & np.isfinite(fwd).all(axis=2)
    valid &= finite

    meta = {
        "geometry_model": model,
        "homography_inlier_ratio": float(hr),
        "fundamental_inlier_ratio": float(fr),
        "residual_threshold_px": float(thr),
    }
    return score.astype(np.float32), valid, meta

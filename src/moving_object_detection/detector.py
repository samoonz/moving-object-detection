from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class MaskConfig:
    min_component_area_px: int = 10
    min_component_area_ratio: float = 5e-6
    morph_open: int = 3
    morph_close: int = 7
    dilate: int = 3
    bbox_padding: int = 4
    min_component_score: float = 1.05


def _odd_kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def build_motion_mask(
    score: np.ndarray,
    valid: np.ndarray,
    cfg: MaskConfig,
) -> Tuple[np.ndarray, List[Dict[str, float | int]]]:
    raw = ((score >= 1.0) & valid).astype(np.uint8) * 255

    if cfg.morph_open > 1:
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, _odd_kernel(cfg.morph_open), iterations=1)
    if cfg.morph_close > 1:
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, _odd_kernel(cfg.morph_close), iterations=1)
    if cfg.dilate > 1:
        raw = cv2.dilate(raw, _odd_kernel(cfg.dilate), iterations=1)

    h, w = raw.shape
    min_area = max(cfg.min_component_area_px, int(h * w * cfg.min_component_area_ratio))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)

    clean = np.zeros_like(raw)
    components: List[Dict[str, float | int]] = []

    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or bw < 2 or bh < 2:
            continue

        comp = labels == i
        comp_score = float(np.median(score[comp])) if np.any(comp) else 0.0
        if comp_score < cfg.min_component_score:
            continue

        clean[comp] = 255
        pad = int(cfg.bbox_padding)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w - 1, x + bw - 1 + pad)
        y2 = min(h - 1, y + bh - 1 + pad)
        components.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "area": area,
                "score": comp_score,
            }
        )

    return clean, components

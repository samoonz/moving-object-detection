#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moving_object_detection.detector import MaskConfig, build_motion_mask
from moving_object_detection.geometry import GeometryConfig, residual_motion_score

h, w = 360, 640
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

fwd = np.zeros((h, w, 2), np.float32)
fwd[..., 0] = 3.0 + 0.002 * yy
fwd[..., 1] = -2.0 + 0.001 * xx

obj = (slice(130, 220), slice(250, 360))
fwd[obj][..., 0] += 12.0
fwd[obj][..., 1] += 4.0

bwd = -fwd.copy()

score, valid, meta = residual_motion_score(
    fwd, bwd, GeometryConfig(fb_consistency_max_px=30.0)
)
mask, comps = build_motion_mask(score, valid, MaskConfig(min_component_area_px=20))

obj_pixels = int(np.count_nonzero(mask[obj]))
all_pixels = int(np.count_nonzero(mask))
recall = obj_pixels / float((220 - 130) * (360 - 250))
precision_proxy = obj_pixels / float(max(all_pixels, 1))

print("geometry:", meta)
print("components:", comps)
print("recall:", round(recall, 3), "precision_proxy:", round(precision_proxy, 3))

if recall < 0.60:
    raise SystemExit("Smoke test failed: moving region was not recovered")
print("Smoke test passed")

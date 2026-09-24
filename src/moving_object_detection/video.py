from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from .detector import MaskConfig, build_motion_mask
from .fast_motion import FastConfig, FastMotionEstimator
from .flow import FlowConfig, SEAFlowEstimator, auto_batch_pairs
from .geometry import GeometryConfig, residual_motion_score


def resize_for_analysis(frame: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if max_side <= 0 or longest <= max_side:
        return frame, 1.0
    scale = max_side / float(longest)
    out = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return out, scale


def _scale_component(comp: Dict[str, Any], inv_scale: float, w: int, h: int) -> Dict[str, Any]:
    out = dict(comp)
    for k in ("x1", "x2"):
        out[k] = int(np.clip(round(float(comp[k]) * inv_scale), 0, w - 1))
    for k in ("y1", "y2"):
        out[k] = int(np.clip(round(float(comp[k]) * inv_scale), 0, h - 1))
    out["area"] = int(round(float(comp["area"]) * inv_scale * inv_scale))
    return out


def process_video(input_path: str, output_dir: str, cfg: Dict[str, Any]) -> List[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_cfg = cfg["video"]
    flow_cfg_raw = cfg["flow"]
    backend = str(flow_cfg_raw.get("backend", "fast")).lower()
    geometry_cfg = GeometryConfig(**cfg.get("geometry", {}))
    mask_cfg = MaskConfig(**cfg["mask"])

    estimator = None
    fast_estimator = None

    if backend in {"fast", "fast_homography", "hybrid_fast"}:
        fast_estimator = FastMotionEstimator(FastConfig(**cfg.get("fast", {})))
    elif backend in {"sea_raft", "sea-raft", "raft"}:
        estimator = SEAFlowEstimator(
            FlowConfig(
                model=flow_cfg_raw["model"],
                checkpoint=flow_cfg_raw["checkpoint"],
                precision=cfg.get("precision", "fp16"),
                allow_tf32=cfg.get("allow_tf32", True),
                torch_compile=cfg.get("torch_compile", False),
                bidirectional=flow_cfg_raw.get("bidirectional", True),
                cudnn_grid_sample_workaround=flow_cfg_raw.get(
                    "cudnn_grid_sample_workaround", True
                ),
                avoid_cudnn_fallback=flow_cfg_raw.get(
                    "avoid_cudnn_fallback", True
                ),
            ),
            device=cfg.get("device", "cuda"),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    stem = Path(input_path).stem
    overlay_path = output_root / f"{stem}_motion_overlay.mp4"
    mask_path = output_root / f"{stem}_motion_mask.mp4"
    jsonl_path = output_root / f"{stem}_detections.jsonl"
    summary_path = output_root / f"{stem}_summary.json"

    codec = cv2.VideoWriter_fourcc(*video_cfg.get("codec", "mp4v"))
    overlay_writer = cv2.VideoWriter(str(overlay_path), codec, fps, (width, height))
    mask_writer = cv2.VideoWriter(str(mask_path), codec, fps, (width, height), isColor=True)
    log_fp = jsonl_path.open("w", encoding="utf-8")

    ret, prev_orig = cap.read()
    if not ret:
        raise RuntimeError("Video contains no frames")

    max_side = int(flow_cfg_raw.get("analysis_max_side", 1280))
    prev_ana, scale = resize_for_analysis(prev_orig, max_side)

    if backend in {"fast", "fast_homography", "hybrid_fast"}:
        batch_pairs = 1
        print(
            f"Backend: FAST sparse LK + multi-homography | "
            f"analysis={prev_ana.shape[1]}x{prev_ana.shape[0]}"
        )
    else:
        batch_pairs = flow_cfg_raw.get("batch_pairs", "auto")
        if batch_pairs == "auto":
            batch_pairs = auto_batch_pairs(
                prev_ana.shape[:2],
                bidirectional=flow_cfg_raw.get("bidirectional", True),
                precision=cfg.get("precision", "fp16"),
                avoid_cudnn_fallback=flow_cfg_raw.get("avoid_cudnn_fallback", True),
            )
        batch_pairs = max(1, int(batch_pairs))
        feature_tokens = ((prev_ana.shape[0] + 7) // 8) * ((prev_ana.shape[1] + 7) // 8)
        effective_grid_batch = (
            batch_pairs
            * feature_tokens
            * (2 if flow_cfg_raw.get("bidirectional", True) else 1)
        )
        print(
            f"Backend: SEA-RAFT | analysis={prev_ana.shape[1]}x{prev_ana.shape[0]} | "
            f"batch_pairs={batch_pairs} | "
            f"bidirectional={flow_cfg_raw.get('bidirectional', True)} | "
            f"grid_batch~{effective_grid_batch:,}"
        )

    overlay_writer.write(prev_orig)
    mask_writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    log_fp.write(json.dumps({"frame": 0, "boxes": [], "geometry": None}) + "\n")

    frame_index = 1
    processed = 0
    geometry_counts: Dict[str, int] = {}
    start = time.perf_counter()

    total_pairs = max(total_frames - 1, 0) if total_frames > 0 else None
    progress = tqdm(
        total=total_pairs,
        desc="Processing",
        unit="frame",
        dynamic_ncols=True,
        mininterval=float(video_cfg.get("progress_mininterval", 0.5)),
        disable=not bool(video_cfg.get("show_progress", True)),
    )

    try:
        while True:
            items = []
            for _ in range(batch_pairs):
                ret, current_orig = cap.read()
                if not ret:
                    break
                current_ana, current_scale = resize_for_analysis(current_orig, max_side)
                if abs(current_scale - scale) > 1e-6:
                    scale = current_scale
                items.append((frame_index, prev_ana, current_ana, current_orig))
                prev_ana = current_ana
                frame_index += 1

            if not items:
                break

            stage_start = time.perf_counter()

            fast_results = None
            fwd_list = None
            bwd_list = None

            if backend in {"fast", "fast_homography", "hybrid_fast"}:
                fast_results = [
                    fast_estimator.estimate(a, b)
                    for _, a, b, _ in items
                ]
            else:
                pairs = [(a, b) for _, a, b, _ in items]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                fwd_list, bwd_list = estimator.estimate_pairs(pairs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            stage_seconds = time.perf_counter() - stage_start

            for j, (idx, _a, _b, current_orig) in enumerate(items):
                if fast_results is not None:
                    score, valid, meta = fast_results[j]
                else:
                    bwd = None if bwd_list is None else bwd_list[j]
                    score, valid, meta = residual_motion_score(
                        fwd_list[j], bwd, geometry_cfg
                    )

                mask_small, comps = build_motion_mask(score, valid, mask_cfg)

                inv_scale = 1.0 / scale
                boxes = [
                    _scale_component(c, inv_scale, width, height)
                    for c in comps
                ]
                mask_full = cv2.resize(
                    mask_small,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

                overlay = current_orig.copy()
                for box in boxes:
                    cv2.rectangle(
                        overlay,
                        (box["x1"], box["y1"]),
                        (box["x2"], box["y2"]),
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        overlay,
                        f"motion {box['score']:.2f}",
                        (box["x1"], max(18, box["y1"] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    overlay,
                    f"{meta['geometry_model']} thr={meta['residual_threshold_px']:.1f}",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                overlay_writer.write(overlay)
                mask_writer.write(cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR))
                log_fp.write(
                    json.dumps(
                        {"frame": idx, "boxes": boxes, "geometry": meta},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                geometry_counts[meta["geometry_model"]] = (
                    geometry_counts.get(meta["geometry_model"], 0) + 1
                )
                processed += 1

            progress.update(len(items))
            elapsed_now = max(time.perf_counter() - start, 1e-9)
            postfix = {
                "fps": f"{processed / elapsed_now:.2f}",
                "stage_s": f"{stage_seconds:.2f}",
                "batch": len(items),
            }
            if backend not in {"fast", "fast_homography", "hybrid_fast"} and torch.cuda.is_available():
                postfix["vram"] = f"{torch.cuda.memory_allocated() / 1024**3:.1f}G"
                postfix["peak"] = f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}G"
            progress.set_postfix(postfix)

    finally:
        progress.close()
        cap.release()
        overlay_writer.release()
        mask_writer.release()
        log_fp.close()

    elapsed = time.perf_counter() - start
    summary = {
        "input": input_path,
        "backend": backend,
        "resolution": [width, height],
        "fps": fps,
        "frames_reported": total_frames,
        "frames_processed": processed + 1,
        "elapsed_seconds": elapsed,
        "processing_fps": processed / elapsed if elapsed > 0 else 0.0,
        "analysis_scale": scale,
        "batch_pairs": batch_pairs,
        "geometry_model_counts": geometry_counts,
        "config": cfg,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return [overlay_path, mask_path, jsonl_path, summary_path]

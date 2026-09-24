from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FlowConfig:
    model: str = "sea_raft_m"
    checkpoint: str = "mixed"
    precision: str = "fp16"
    allow_tf32: bool = True
    torch_compile: bool = False
    bidirectional: bool = True


class SEAFlowEstimator:
    """PTLFlow-backed SEA-RAFT inference optimized for batched video pairs."""

    def __init__(self, cfg: FlowConfig, device: str = "cuda") -> None:
        import ptlflow

        self.cfg = cfg
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for the A100 profile.")

        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)
        torch.backends.cudnn.benchmark = True

        self.model = ptlflow.get_model(cfg.model, cfg.checkpoint).to(self.device).eval()
        if hasattr(self.model, "args") and hasattr(self.model.args, "use_tile_input"):
            self.model.args.use_tile_input = False

        if cfg.torch_compile:
            self.model = torch.compile(self.model, mode="reduce-overhead")

        if cfg.precision == "bf16":
            self.autocast_dtype = torch.bfloat16
        elif cfg.precision == "fp16":
            self.autocast_dtype = torch.float16
        else:
            self.autocast_dtype = None

    @staticmethod
    def _to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        return t

    @staticmethod
    def _pad8(images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h, w = images.shape[-2:]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        if ph or pw:
            images = F.pad(images, (0, pw, 0, ph), mode="replicate")
        return images, (h, w)

    def _infer_tensor(self, images: torch.Tensor) -> torch.Tensor:
        images, (h, w) = self._pad8(images)
        images = images.to(self.device, non_blocking=True)

        with torch.inference_mode():
            if self.autocast_dtype is None:
                pred = self.model({"images": images})
            else:
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    pred = self.model({"images": images})

        flows = pred["flows"][:, 0, :, :h, :w]
        return flows.float()

    def _infer_with_oom_fallback(self, images: torch.Tensor) -> torch.Tensor:
        try:
            return self._infer_tensor(images)
        except torch.cuda.OutOfMemoryError:
            if images.shape[0] <= 1:
                raise
            torch.cuda.empty_cache()
            mid = images.shape[0] // 2
            a = self._infer_with_oom_fallback(images[:mid])
            b = self._infer_with_oom_fallback(images[mid:])
            return torch.cat([a, b], dim=0)

    def estimate_pairs(
        self,
        pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray] | None]:
        if not pairs:
            return [], [] if self.cfg.bidirectional else None

        forward_items = []
        backward_items = []
        for a, b in pairs:
            ta = self._to_tensor(a)
            tb = self._to_tensor(b)
            forward_items.append(torch.stack([ta, tb], dim=0))
            if self.cfg.bidirectional:
                backward_items.append(torch.stack([tb, ta], dim=0))

        all_items = forward_items + backward_items
        batch = torch.stack(all_items, dim=0)
        flows = self._infer_with_oom_fallback(batch).cpu().numpy()

        n = len(forward_items)
        fwd = [flows[i].transpose(1, 2, 0).copy() for i in range(n)]
        if not self.cfg.bidirectional:
            return fwd, None
        bwd = [flows[n + i].transpose(1, 2, 0).copy() for i in range(n)]
        return fwd, bwd


def auto_batch_pairs() -> int:
    if not torch.cuda.is_available():
        return 1
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_gb >= 70:
        return 4
    if total_gb >= 35:
        return 2
    return 1

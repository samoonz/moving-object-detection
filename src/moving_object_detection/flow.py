from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F


SEA_RAFT_CHECKPOINTS = ("tartan", "chairs", "things", "sintel", "kitti", "spring")
SEA_RAFT_DEFAULT_CHECKPOINT = "spring"


@dataclass
class FlowConfig:
    model: str = "sea_raft_m"
    checkpoint: str = SEA_RAFT_DEFAULT_CHECKPOINT
    precision: str = "fp16"
    allow_tf32: bool = True
    torch_compile: bool = False
    bidirectional: bool = True
    cudnn_grid_sample_workaround: bool = True
    avoid_cudnn_fallback: bool = True


def _install_safe_grid_sample() -> None:
    """
    Work around PyTorch/cuDNN grid_sample failures for very large effective
    batch dimensions used by SEA-RAFT's all-pairs correlation sampler.

    PyTorch issue #88380 documents CUDNN_STATUS_NOT_SUPPORTED once the
    grid_sample batch dimension becomes very large. Making tensors contiguous
    alone is not sufficient. We keep cuDNN enabled globally and disable it
    only for the affected grid_sample call.
    """
    if getattr(F.grid_sample, "_mod_safe_grid_sample", False):
        return

    original_grid_sample = F.grid_sample

    def safe_grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
        input_c = input.contiguous()
        grid_c = grid.contiguous()

        # SEA-RAFT reshapes correlation to [B*H*W, C, h, w].
        # cuDNN grid_sample is known to fail at/above ~65k leading batches.
        if input_c.is_cuda and input_c.shape[0] >= 65536:
            with torch.backends.cudnn.flags(enabled=False):
                return original_grid_sample(
                    input_c,
                    grid_c,
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )

        return original_grid_sample(
            input_c,
            grid_c,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    safe_grid_sample._mod_safe_grid_sample = True
    F.grid_sample = safe_grid_sample


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

        if cfg.cudnn_grid_sample_workaround:
            _install_safe_grid_sample()

        checkpoint = cfg.checkpoint
        try:
            self.model = ptlflow.get_model(cfg.model, checkpoint)
        except ValueError as exc:
            message = str(exc)
            if "Invalid checkpoint name" not in message:
                raise
            warnings.warn(
                f"PTLFlow rejected checkpoint {checkpoint!r}. "
                f"Falling back to {SEA_RAFT_DEFAULT_CHECKPOINT!r}. "
                f"Known SEA-RAFT checkpoints: {', '.join(SEA_RAFT_CHECKPOINTS)}"
            )
            checkpoint = SEA_RAFT_DEFAULT_CHECKPOINT
            self.model = ptlflow.get_model(cfg.model, checkpoint)

        self.checkpoint = checkpoint
        self.model = self.model.to(self.device).eval()
        print(f"Loaded PTLFlow model={cfg.model} checkpoint={self.checkpoint}")

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


def auto_batch_pairs(
    frame_hw: tuple[int, int] | None = None,
    bidirectional: bool = True,
    precision: str = "fp16",
    avoid_cudnn_fallback: bool = True,
) -> int:
    """
    Pick a batch size from both VRAM and analysis resolution.

    SEA-RAFT's default CorrBlock materializes all-pairs correlation, so memory
    grows roughly with (H/8 * W/8)^2. A100 capacity alone is therefore not a
    sufficient batch-size heuristic.
    """
    if not torch.cuda.is_available():
        return 1

    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    hard_cap = 4 if total_gb >= 70 else 2 if total_gb >= 35 else 1

    if frame_hw is None:
        return hard_cap

    h, w = frame_hw
    h8 = (int(h) + 7) // 8
    w8 = (int(w) + 7) // 8
    tokens = h8 * w8

    bytes_per_value = 4 if precision == "fp32" else 2
    pyramid_factor = 1.34
    directions = 2 if bidirectional else 1

    corr_bytes_per_pair = (
        directions
        * tokens
        * tokens
        * bytes_per_value
        * pyramid_factor
    )

    # Leave ample room for feature maps, update blocks, decoder state,
    # optical-flow outputs and CUDA allocator fragmentation.
    if total_gb >= 70:
        corr_budget_gb = 24.0
    elif total_gb >= 35:
        corr_budget_gb = 12.0
    else:
        corr_budget_gb = 4.0

    budget_bytes = corr_budget_gb * (1024 ** 3)
    by_resolution = max(1, int(budget_bytes // max(corr_bytes_per_pair, 1)))

    # PTLFlow SEA-RAFT CorrBlock reshapes its cost volume to
    # [model_batch * H8 * W8, C, h, w] before grid_sample. PyTorch/cuDNN
    # grid_sample is problematic once that leading dimension reaches ~65536.
    # Staying below the limit is usually MUCH faster than taking the
    # non-cuDNN fallback, even on an A100 with plenty of VRAM.
    if avoid_cudnn_fallback:
        directions = 2 if bidirectional else 1
        cudnn_cap = max(1, 65535 // max(tokens * directions, 1))
    else:
        cudnn_cap = hard_cap

    return max(1, min(hard_cap, by_resolution, cudnn_cap))

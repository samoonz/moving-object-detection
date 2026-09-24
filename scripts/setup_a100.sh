#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip setuptools wheel

if python3 - <<'PY'
try:
    import torch
    print("Existing torch:", torch.__version__, "CUDA:", torch.version.cuda)
except Exception:
    raise SystemExit(1)
PY
then
  echo "Using existing PyTorch."
else
  echo "PyTorch not found; installing CUDA 12.4 wheel."
  python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

python3 -m pip install -r requirements.txt
python3 -m pip install -e .

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("gpu:", torch.cuda.get_device_name(0))
print("vram GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY

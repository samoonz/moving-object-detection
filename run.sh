#!/usr/bin/env bash
set -euo pipefail
python -m moving_object_detection.cli --config configs/a100.yaml "$@"

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gdown
import yaml

from .drive import upload_files
from .video import process_video


def parse_args():
    p = argparse.ArgumentParser(description="Moving-object detection robust to camera motion")
    p.add_argument("--config", default="configs/a100.yaml")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--input", help="Local video path")
    src.add_argument("--drive-url", help="Public Google Drive video URL")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--upload", action="store_true", help="Upload outputs to configured Drive folder")
    return p.parse_args()


def _download_drive_file(url: str, output: str) -> str:
    """
    Download a public Google Drive file with current gdown APIs.

    gdown >= 6 removed the old fuzzy=True argument and now extracts the
    file ID from standard Google Drive share links automatically.
    """
    try:
        result = gdown.download(url=url, output=output, quiet=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download input video from Google Drive: {exc}"
        ) from exc

    if not result:
        raise RuntimeError("Failed to download input video from Google Drive")
    return str(result)


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    input_path = args.input
    drive_url = args.drive_url or cfg.get("drive", {}).get("input_url")
    if not input_path:
        if not drive_url:
            raise SystemExit("Provide --input or --drive-url (or drive.input_url in config).")
        cache_dir = Path("data")
        cache_dir.mkdir(parents=True, exist_ok=True)
        input_path = str(cache_dir / "input_video.mp4")
        cached = Path(input_path)
        if cached.exists() and cached.stat().st_size > 0:
            print(f"Using cached input video: {cached} ({cached.stat().st_size / 1024**3:.2f} GiB)")
        else:
            input_path = _download_drive_file(drive_url, input_path)

    outputs = process_video(input_path, args.output_dir, cfg)
    print("\nOutputs:")
    for p in outputs:
        print(" -", p)

    should_upload = args.upload or bool(cfg.get("drive", {}).get("auto_upload", False))
    if should_upload:
        folder_id = cfg["drive"]["output_folder_id"]
        drive_cfg = cfg.get("drive", {})
        ids = upload_files(
            outputs,
            folder_id,
            method=drive_cfg.get("upload_method", "rclone"),
            rclone_remote=drive_cfg.get("rclone_remote", "gdrive"),
        )
        print("Uploaded Drive file IDs:")
        print(json.dumps(ids, indent=2))


if __name__ == "__main__":
    main()

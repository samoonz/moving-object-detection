from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List


def _upload_rclone(paths: list[Path], folder_id: str, remote: str) -> List[str]:
    if shutil.which("rclone") is None:
        raise RuntimeError(
            "rclone is not installed. Install it and run `rclone config` once, "
            "or choose drive.upload_method=service_account."
        )

    uploaded: List[str] = []
    remote = remote.rstrip(":") + ":"
    for path in paths:
        cmd = [
            "rclone",
            "copyto",
            str(path),
            f"{remote}{path.name}",
            "--drive-root-folder-id",
            folder_id,
            "--progress",
        ]
        subprocess.run(cmd, check=True)
        uploaded.append(path.name)
    return uploaded


def _upload_service_account(paths: list[Path], folder_id: str) -> List[str]:
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is not set. For service-account upload, "
            "point it to a JSON key and share the destination Shared Drive/folder with that account."
        )

    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    uploaded: List[str] = []
    for path in paths:
        media = MediaFileUpload(str(path), resumable=True)
        body = {"name": path.name, "parents": [folder_id]}
        item = (
            service.files()
            .create(body=body, media_body=media, fields="id,name", supportsAllDrives=True)
            .execute()
        )
        uploaded.append(item["id"])
    return uploaded


def upload_files(
    paths: Iterable[str | Path],
    folder_id: str,
    method: str = "rclone",
    rclone_remote: str = "gdrive",
) -> List[str]:
    files = [Path(p) for p in paths]
    files = [p for p in files if p.exists() and p.is_file()]
    if not files:
        return []

    method = method.lower().strip()
    if method == "rclone":
        return _upload_rclone(files, folder_id, rclone_remote)
    if method in {"service_account", "service-account"}:
        return _upload_service_account(files, folder_id)
    raise ValueError(f"Unknown Drive upload method: {method}")

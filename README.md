# Moving Object Detection — moving camera / camera shake / A100

Phát hiện **chuyển động độc lập với camera**, không nhận dạng class. Project dành cho video có rung camera, pan/tilt/rotation, thay đổi góc nhìn và vật thể thay đổi kích thước mạnh.

## Thiết kế mặc định

Pipeline:

1. **SEA-RAFT** (`sea_raft_m`, pretrained `mixed`) tính dense optical flow trên GPU.
2. Chạy forward + backward flow cùng batch để tăng utilization A100 và lọc flow lỗi.
3. Lấy dense correspondences từ optical flow.
4. Fit đồng thời:
   - Homography + MAGSAC: mạnh khi rung/rotation/zoom hoặc cảnh gần phẳng.
   - Fundamental matrix + MAGSAC: mạnh hơn homography khi camera tịnh tiến trong cảnh 3D có parallax.
5. Tự chọn mô hình camera có support tốt hơn.
6. Tính geometric residual cho từng pixel.
7. Adaptive threshold bằng median/MAD, forward-backward consistency, morphology và connected components.
8. Xuất bounding boxes, mask video, overlay video và JSONL.

Điểm chính: background không được xác định bằng pixel difference/MOG2. Chuyển động camera được mô hình hóa trước, sau đó chỉ giữ phần **residual motion** không giải thích được bởi camera.

### Vì sao SEA-RAFT thay vì WAFT làm mặc định?

WAFT (ICLR 2026) mới hơn và có kết quả benchmark rất mạnh. Tuy nhiên repo public hiện ít ổn định hơn cho custom downstream inference. SEA-RAFT (ECCV 2024) đã có model zoo, HuggingFace và PTLFlow integration, phù hợp hơn cho một pipeline cần chạy chắc chắn trên server. Backend được tách riêng để sau này thay SEA-RAFT bằng WAFT mà không đổi geometry/mask pipeline.

References:
- SEA-RAFT: https://github.com/princeton-vl/SEA-RAFT
- SEA-RAFT paper: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1065_ECCV_2024_paper.php
- WAFT: https://github.com/princeton-vl/WAFT
- WAFT paper: https://openreview.net/forum?id=HTqGE0KcuF

## Cài trên A100

Khuyến nghị Python 3.10, CUDA 12.x.

```bash
git clone https://github.com/samoonz/moving-object-detection.git
cd moving-object-detection
bash scripts/setup_a100.sh
```

Kiểm tra GPU:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.get_device_name(0))
PY
```

## Chạy video mặc định từ Google Drive

`configs/a100.yaml` đã chứa input video:

```text
https://drive.google.com/file/d/1gVcC5ujuMZ8q5Rdp76WezN6xXQFYY2fk/view?usp=sharing
```

Chạy:

```bash
./run.sh
```

Hoặc video local:

```bash
./run.sh --input /path/video.mp4
```

Kết quả nằm trong `outputs/`:

```text
*_motion_overlay.mp4
*_motion_mask.mp4
*_detections.jsonl
*_summary.json
```

## Upload tự động lên Google Drive

Folder đích đã cấu hình:

```text
1GvQhEPQ5aPXQGD6JZZyUSHHRQuNuVS4l
```

### Cách khuyến nghị cho server: rclone

Với folder My Drive thông thường, `rclone` ổn định hơn service account vì file được ghi bằng chính tài khoản Google của bạn. Cài `rclone`, chạy một lần:

```bash
rclone config
```

Tạo remote tên `gdrive`, đăng nhập tài khoản Google có quyền Editor với folder đích. Sau đó:

```bash
./run.sh --upload
```

Project dùng `--drive-root-folder-id`, nên không cần biết tên/path của folder; ID trong config được dùng trực tiếp.

### Service account (phù hợp Shared Drive)

Đổi trong `configs/a100.yaml`:

```yaml
drive:
  upload_method: service_account
```

Sau đó:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/secure/gdrive-sa.json
./run.sh --upload
```

Không commit JSON credential lên GitHub. Với My Drive thông thường, ưu tiên `rclone`; service account phù hợp hơn khi folder nằm trong Shared Drive hoặc môi trường đã cấu hình quota/ủy quyền phù hợp.

## Tối ưu A100

Default:

```yaml
precision: fp16
allow_tf32: true
flow:
  model: sea_raft_m
  analysis_max_side: 1920
  bidirectional: true
  batch_pairs: auto
```

`batch_pairs: auto` chọn:
- A100 80 GB: 4 frame-pairs/batch (8 directions khi bidirectional)
- A100 40 GB: 2 frame-pairs/batch
- GPU nhỏ hơn: 1

Nếu OOM, estimator tự chia batch nhỏ hơn.

Nếu cần ưu tiên vật thể cực nhỏ ở video 4K, đổi:

```yaml
flow:
  analysis_max_side: 3840
  batch_pairs: 1
```

Nếu ưu tiên tốc độ:

```yaml
flow:
  analysis_max_side: 1280
  bidirectional: false
```

Nếu muốn thử accuracy cao hơn và VRAM đủ:

```yaml
flow:
  model: sea_raft_l
```

## Vì sao dùng cả Fundamental matrix và Homography?

Homography xử lý camera shake/rotation/zoom rất tốt nhưng có thể tạo false positive khi cảnh 3D có parallax. Fundamental matrix không giả định toàn cảnh nằm trên một mặt phẳng nên phù hợp hơn khi camera tịnh tiến và các vật nền ở nhiều độ sâu. Pipeline fit cả hai bằng robust MAGSAC rồi chọn mô hình có inlier support phù hợp.

## Giới hạn cần biết

- Một vật thể đứng yên tương đối với cảnh sẽ không bị đánh dấu — đúng với mục tiêu "chỉ bắt chuyển động".
- Motion dọc chính xác theo epipolar line có thể khó hơn cho Fundamental residual; Homography/fallback và temporal continuity giúp giảm trường hợp này.
- Motion blur cực mạnh hoặc vật thể chỉ vài pixel vẫn phụ thuộc chất lượng optical flow; tăng `analysis_max_side` khi cần.
- Đây là motion segmentation, không phải semantic detector. Không cần train class và không phụ thuộc vật thể là người, xe, drone hay vật thể lạ.

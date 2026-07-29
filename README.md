# TempoRun 2026 Reproduce

## Thông tin nộp bài

- Tên đội: `Tôi thăng cấp một mình`
- Tên phương pháp: `TempoRun 2026 OmniShotCut + Qwen3-VL Temporal Reproduce`
- Đường dẫn repository: `https://github.com/Feph50/Temporun2026`
- Phương thức cài đặt môi trường: `Docker`
- Cấu hình phần cứng đã kiểm thử: `RTX 2080 Ti 11GB, Python 3.10.15, Ubuntu 22.04, CUDA 12.1 devel`
- Lệnh chạy chính: xem mục 8 và 9
- Thông tin liên hệ: `0886920075`

## 1. Giới thiệu ngắn gọn về phương pháp

Pipeline này dùng OmniShotCut để tách shot, lấy keyframe giữa shot, sinh thêm extra frames, tạo embedding bằng Qwen3-VL-Embedding-8B, sau đó retrieve và rerank bằng Qwen3-VL-Reranker-8B. Kết quả cuối được tinh chỉnh thêm bằng notebook temporal rerank.

## 2. Mô tả cấu trúc repository

| File | Vai trò |
| --- | --- |
| `stage1_extract_keyframe.py` | Trích xuất middle frame/keyframe của từng shot. |
| `stage1_extract_extra.py` | Trích xuất thêm extra frames quanh các range/keyframe đã có. |
| `stage2_embedding_frames.py` | Tạo embedding `.pt` cho keyframes và merge các shard embedding. |
| `merge_temporun_keyframes_windows.py` | Gộp folder middle keyframes và extra keyframes thành một folder combined. |
| `stage3_retrieve_and_reranker.py` | Retrieve candidates và rerank kết quả. |
| `finalStage_temporal_frames.ipynb` | Notebook final stage để temporal rerank/top-5 rerank. |
| `OmniShotCut/` | Source repo OmniShotCut, dùng để import package `omnishotcut/...`. |
| `requirements.txt` | Danh sách Python packages cần cài thêm. |

## 3. Yêu cầu phần cứng và phần mềm

- GPU tối thiểu khuyến nghị: `RTX 2080 Ti 11GB`
- Nếu có nhiều GPU, code đã được thiết kế để chạy song song và tận dụng đa GPU tốt hơn.
- Môi trường đã kiểm tra: Ubuntu 22.04, Python 3.10.15, CUDA 12.1 devel runtime, NVIDIA driver 580.82.09, CUDA 13.0 trên host
- `torch`, `torchvision`
- `ffmpeg` binary trên `PATH`
- CUDA-compatible environment
- `ffmpeg-python` chỉ là wrapper, vẫn cần binary `ffmpeg` thật để decode video.

Khi chạy các lệnh stage 1, 2, 3 và notebook, checkpoint `Qwen/Qwen3-VL-Embedding-8B` và `Qwen/Qwen3-VL-Reranker-8B` sẽ tự động được tải từ Hugging Face nếu chưa có sẵn local.

## 4. Hướng dẫn cài đặt môi trường

### Docker

Đây là đường chạy chính. Chạy từ thư mục `Temporun2026`:

```bash
docker build -t temporun2026:latest .

docker run --gpus all -it --rm \
  -v .:/workspace/Temporun2026 \
  -w /workspace/Temporun2026 \
  temporun2026:latest
```

### Cài tay ngoài Docker

Chỉ dùng khi không chạy bằng Docker:

```bash
pip install -r requirements.txt
```

Lưu ý: `ffmpeg-python` là Python wrapper. Môi trường chạy vẫn cần có binary `ffmpeg` nếu dùng các hàm decode video của OmniShotCut.

## 5. Hướng dẫn tải checkpoint hoặc tài nguyên bổ sung

Không cần tải tay checkpoint embedding/reranker trước. Khi chạy:

- `stage1_extract_keyframe.py`
- `stage1_extract_extra.py`
- `stage2_embedding_frames.py`
- `stage3_retrieve_and_reranker.py`
- `finalStage_temporal_frames.ipynb`

thì checkpoint `Qwen/Qwen3-VL-Embedding-8B` và `Qwen/Qwen3-VL-Reranker-8B` sẽ tự động tải từ Hugging Face nếu chưa có local cache.

Tài nguyên bổ sung cần có sẵn:

- video corpus `V3C`
- file `private_round_tasks.jsonl`

## 6. Mô tả dữ liệu đầu vào

Stage 1 nhận dữ liệu video qua tham số `--dataset-root`.

Cấu trúc dữ liệu video dự kiến:

```text
dataset/
├── V3C1/
│   └── videos/
│       └── ...
└── V3C2/
    └── videos/
        └── ...
```

Các script stage sau dùng:

- folder keyframes đã sinh ra
- `private_round_tasks.jsonl`
- index `.pt`
- folder `V3C`
- folder model `Qwen3-VL-Reranker-8B`

## 7. Mô tả kết quả đầu ra

Khi chạy xong từng phần, code sẽ tạo ra các đầu ra sau:

- Stage 1:
  - `TempoRun2026_OmniShotCut_Keyframes/`
  - `TempoRun2026_OmniShotCut_Extra_Keyframes_More/`
  - mỗi video có `k_*.jpg`, `ts_ms.npy`, và file metadata tương ứng
- Stage 2:
  - `checkpoints_part/`
  - `qwen3vl_final_full_index_8B.pt`
- Stage 3:
  - `retrieval_candidates_8B.json`
  - `submission_final_8B_rerank500_10.json`
- Final stage notebook:
  - `temporal_top5_rerank_final_private_task/`
  - `submission_final_temporal_top5_private.json`
  - `submission_final_temporal_top5_private.zip`
  - `temporal_top5_scores.csv`

Output JSON cuối phải có khóa ngoài cùng là `predictions` và giữ đúng `task_id`, `rank`, `video_id`, `frame_ms`.

## 8. Hướng dẫn chạy từng script

### Stage 1

```bash
# extract middle frames of each shot.
python3 stage1_extract_keyframe.py \
--dataset-root "V3C/V3C1" \
--dataset-root "V3C/V3C2" \
--out "TempoRun2026_OmniShotCut_Keyframes"

# extract extra frames
python3 stage1_extract_extra.py \
--ranges-root "TempoRun2026_OmniShotCut_Keyframes" \
--out "TempoRun2026_OmniShotCut_Extra_Keyframes_More" \
--candidate-fps 2 \
--cosine-threshold 0.97 \
--output-mode extras_only \
--filename-digits 6 \
--batch-size 128 \
--jpeg-quality 92 \
--device cuda
```

### Stage 2

```bash
python3 stage2_embedding_frames.py embed-pt  \
--keyframes TempoRun2026_OmniShotCut_Keyframes   \
--checkpoint-dir checkpoints_part  \
--load-in-4bit   \
--batch-size 32

python3 stage2_embedding_frames.py embed-pt  \
--keyframes "TempoRun2026_OmniShotCut_Extra_Keyframes_More"   \
--checkpoint-dir checkpoints_part  \
--load-in-4bit   \
--batch-size 32

# Merge cac file .pt cho mọi video
python3 stage2_embedding_frames.py merge \
--checkpoint-dir checkpoints_part \
--out qwen3vl_final_full_index_8B.pt
```

### Stage 3

Trước khi chạy, hãy sửa tên `1_CausalScoreHead` bên trong `modules.json` của `Qwen3vl-Reranker-8B` thành `1_LogitScore`.

```bash
# Gộp 2 folder keyframes lại.
python3 merge_temporun_keyframes_windows.py

python3 stage3_retrieve_and_reranker.py retrieve  \
--model-id "Qwen/Qwen3-VL-Embedding-8B"   \
--index-pt "qwen3vl_final_full_index_8B.pt"  \
--tasks "private_round_tasks.jsonl"   \
--load-in-4bit  \
--top-videos 200  \
--frames-per-video 257  \
--out "retrieval_candidates_8B.json"

python3 stage3_retrieve_and_reranker.py rerank  \
--model-id "Qwen/Qwen3-VL-Reranker-8B"  \
--candidates "retrieval_candidates_8B.json"  \
--tasks "private_round_tasks.jsonl"  \
--index-pt "qwen3vl_final_full_index_8B.pt"  \
--keyframes "TempoRun2026_OmniShotCut_Keyframes_Combined" \
--rerank-top 500  \
--final-top 10   \
--out "submission_final_8B_rerank500_10.json"
```

### Final stage notebook

Lấy output là file `submission_final_8B_rerank500_10.json` để nạp vào file `finalStage_temporal_frames.ipynb`. Notebook sẽ tạo thêm JSON nộp cuối, file ZIP tương ứng và CSV chẩn đoán trong `temporal_top5_rerank_final_private_task/`.

## 9. Lệnh chạy toàn bộ pipeline

Chuỗi chạy đầy đủ:

1. `stage1_extract_keyframe.py`
2. `stage1_extract_extra.py`
3. `stage2_embedding_frames.py embed-pt` cho middle keyframes
4. `stage2_embedding_frames.py embed-pt` cho extra keyframes
5. `stage2_embedding_frames.py merge`
6. `merge_temporun_keyframes_windows.py`
7. `stage3_retrieve_and_reranker.py retrieve`
8. `stage3_retrieve_and_reranker.py rerank`
9. `finalStage_temporal_frames.ipynb`

## 10. Các tham số mặc định

Một số tham số mặc định chính:

- `stage1_extract_keyframe.py`
  - `--device`: `cuda` nếu có GPU, ngược lại `cpu`
  - `--mode`: `default`
  - `--overlap-window-length`: `20`
  - `--shard-index`: `0`
  - `--shard-count`: `1`

- `stage1_extract_extra.py`
  - `--candidate-fps`: `1.0`
  - `--cosine-threshold`: `0.98`
  - `--output-mode`: `extras_only`
  - `--filename-digits`: `5`
  - `--jpeg-quality`: `92`
  - `--batch-size`: `128`
  - `--device`: `cuda` nếu có GPU, ngược lại `cpu`

- `stage2_embedding_frames.py`
  - `--model-id`: `Qwen/Qwen3-VL-Embedding-8B`
  - `--checkpoint-dir`: `checkpoints_part`
  - `--batch-size`: `16` hoặc `32` tùy subcommand
  - `--load-in-4bit`: tắt mặc định
  - `--truncate-dim`: `1024`
  - `--shard-index`: `0`
  - `--shard-count`: `1`

- `stage3_retrieve_and_reranker.py`
  - `retrieve --top-videos`: `5006`
  - `retrieve --frames-per-video`: `257`
  - `retrieve --cand-keyframes`: `300000`
  - `rerank --rerank-top`: `50`
  - `rerank --final-top`: `10`

- `finalStage_temporal_frames.ipynb`
  - `top_k`: `5`
  - `final_top`: `10`
  - `max_gpus`: `2`
  - `limit_tasks`: `0`

## 11. Các lỗi hoặc giới hạn đã biết

- Repo phụ thuộc vào CUDA GPU để chạy đúng tốc độ và để load các model 8B.
- Nếu không có đủ VRAM, cần bật quantization `4bit`/`8bit` như các lệnh mẫu đang dùng.
- `ffmpeg` binary phải có sẵn trên máy.
- Đường dẫn dataset phải đúng với mount thực tế; nếu mount khác `/V3C` thì cần đổi theo môi trường chạy.
- `modules.json` của `Qwen3-VL-Reranker-8B` có thể cần tên module tương thích như README đã ghi.
- Notebook final stage yêu cầu đủ `submission_final_8B_rerank500_10.json`, `private_round_tasks.jsonl`, `V3C/` và `Qwen3-VL-Reranker-8B/` trong working directory.
- Nếu video path lưu trong metadata bị stale, stage 1/extra có fallback scan dataset nhưng vẫn cần corpus truy cập được.

## Cần BTC sửa cụ thể

- Cần chạy từ thư mục repo để các path trong lệnh là tương đối.
- Khi chạy `finalStage_temporal_frames.ipynb`, cần đảm bảo current working directory có đủ `submission_final_8B_rerank500_10.json`, `private_round_tasks.jsonl`, `V3C/` và `Qwen3-VL-Reranker-8B/`.

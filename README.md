# TempoRun 2026 Reproduce

README này mô tả lại thứ tự chạy pipeline trong thư mục `temporun2026_reproduce`.

Các lệnh bên dưới được giữ nguyên theo README gốc. Phần mình chỉnh chỉ là cấu trúc tài liệu, tiêu đề, ghi chú và cách trình bày.

## File chính trong thư mục

| File | Vai trò |
| --- | --- |
| `stage1_extract_keyframe.py` | Trích xuất middle frame/keyframe của từng shot. |
| `stage1_extract_extra.py` | Trích xuất thêm extra frames quanh các range/keyframe đã có. |
| `stage2_embedding_frames.py` | Tạo embedding `.pt` cho keyframes và merge các shard embedding. |
| `merge_temporun_keyframes_windows.py` | Gộp folder middle keyframes và extra keyframes thành một folder combined. |
| `stage2_retrieve_and_reranker.py` | Retrieve candidates và rerank kết quả. |
| `finalStage_temporal_frames.ipynb` | Notebook final stage để temporal rerank/top-5 rerank. |
| `OmniShotCut/` | Source repo OmniShotCut, dùng để import package `omnishotcut/...`. |
| `requirements.txt` | Danh sách Python packages cần cài thêm, không bao gồm `torch` và `torchvision`. |

## Luồng chạy tổng quát

1. Stage 1: tạo middle keyframes và extra keyframes.
2. Stage 2: embed hai folder keyframes, sau đó merge thành index `.pt`.
3. Stage 3: gộp folder keyframes, retrieve candidates, rồi rerank.
4. Final Stage: đưa output JSON vào notebook `finalStage_temporal_frames.ipynb`.

## Cài dependencies

Môi trường chạy cần có sẵn `python3`, `torch` và `torchvision`.

Các package còn lại cài bằng:

```bash
pip install -r requirements.txt
```

Lưu ý: `ffmpeg-python` là Python wrapper. Môi trường chạy vẫn cần có binary `ffmpeg` nếu dùng các hàm decode video của OmniShotCut.

## Stage 1 - Extract frames

Output mong đợi sau stage này:

- `TempoRun2026_OmniShotCut_Keyframes`
- `TempoRun2026_OmniShotCut_Extra_Keyframes_More`

```bash
# extract middle frames of each shot.
python3 stage1_extract_keyframe.py \
  --dataset-root "/V3C/V3C1" \
  --dataset-root "/V3C/V3C2" \
  --out "/TempoRun2026_OmniShotCut_Keyframes"

# extract extra frames
python3 stage1_extract_extra.py \
  --ranges-root "/TempoRun2026_OmniShotCut_Keyframes" \
  --out "/TempoRun2026_OmniShotCut_Extra_Keyframes_More" \
  --candidate-fps 2 \
  --cosine-threshold 0.97 \
  --output-mode extras_only \
  --filename-digits 6 \
  --batch-size 128 \
  --jpeg-quality 92 \
  --device cuda
```

## Stage 2 - Embed keyframes và merge index

Stage này tạo embedding cho:

- Middle keyframes.
- Extra keyframes.

Sau đó merge các file `.pt` thành index cuối.

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

Output mong đợi:

- `checkpoints_part`
- `qwen3vl_final_full_index_8B.pt`

## Stage 3 - Merge keyframes, retrieve và rerank

Trước khi chạy, hãy sửa tên `1_CausalScoreHead` bên trong `modules.json` của `Qwen3vl-Reranker-8B` thành `1_LogitScore`.

```bash
# Gộp 2 folder keyframes lại.
python3 merge_temporun_keyframes_windows.py

python3 stage2_retrieve_and_reranker.py retrieve  \
  --model-id "Qwen3-VL-Embedding-8B"   \
  --index-pt "qwen3vl_final_full_index_8B.pt"  \
  --tasks "private_round_tasks.jsonl"   \
  --load-in-4bit  \
  --top-videos 200  \
  --frames-per-video 257  \
  --out "retrieval_candidates_8B.json"

python3 stage2_retrieve_and_reranker.py rerank  \
  --model-id "Qwen3-VL-Reranker-8B"  \
  --candidates "retrieval_candidates_8B.json"  \
  --tasks "private_round_tasks.jsonl"  \
  --index-pt "qwen3vl_final_full_index_8B.pt"  \
  --keyframes "TempoRun2026_OmniShotCut_Keyframes_Combined" \
  --rerank-top 500  \
  --final-top 10   \
  --out "submission_final_8B_rerank500_10.json"
```

Output mong đợi:

- `retrieval_candidates_8B.json`
- `submission_final_8B_rerank500_10.json`

## Final Stage - Temporal rerank bằng notebook

Lấy output là file `submission_final_8B_rerank500_10.json` để nạp vào file `finalStage_temporal_frames.ipynb`.

Input của notebook chỉ bao gồm:

- File `submission_final_8B_rerank500_10.json`.
- Folder `V3C`.
- File private task.
- Folder model `Qwen3-VL-Reranker-8B`.

Notebook hiện dùng path tương đối:

| Biến trong notebook | Path |
| --- | --- |
| `input_submission` | `submission_final_8B_rerank500_10.json` |
| `tasks_jsonl_candidates` | `private_round_tasks.jsonl` |
| `video_dataset_root` | `V3C` |
| `reranker_model` | `Qwen3-VL-Reranker-8B` |
| `output_dir` | `temporal_top5_rerank_final_private_task` |

Các path tương đối này được resolve theo current working directory của Jupyter kernel, không nhất thiết theo vị trí file `.ipynb`.

Khuyến nghị khi chạy notebook: đặt current working directory là folder repo, cùng cấp với `finalStage_temporal_frames.ipynb`. Khi đó output sẽ được tạo tại:

```text
temporal_top5_rerank_final_private_task/
```

Nếu chạy trên Kaggle và current working directory là `/kaggle/working`, output sẽ nằm tại:

```text
/kaggle/working/temporal_top5_rerank_final_private_task/
```

## Cần BTC xác nhận hoặc sửa cụ thể

Mình đã quét path trong thư mục này và thấy các điểm sau chưa thật rõ:

- Cần xác nhận BTC sẽ mount dataset video thành `/V3C`. Nếu không, các lệnh Stage 1 cần đổi `/V3C/V3C1`, `/V3C/V3C2` sang path thật của folder `V3C`.
- Khi chạy `finalStage_temporal_frames.ipynb`, cần đảm bảo current working directory có đủ `submission_final_8B_rerank500_10.json`, `private_round_tasks.jsonl`, `V3C/` và `Qwen3-VL-Reranker-8B/`.

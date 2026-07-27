import argparse
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_CACHE_DIR = "root/hf_cache"

def list_keyframe_dirs(kf_root: str) -> list[Path]:
    return [Path(p).parent for p in sorted(glob.glob(os.path.join(kf_root, "*", "ts_ms.npy")))]

def load_frame_docs(vdir: Path) -> tuple[list[dict[str, str]], list[int], list[str]]:
    files = sorted(glob.glob(str(vdir / "k_*.jpg")))
    ts = np.load(vdir / "ts_ms.npy")
    n = min(len(files), len(ts))
    docs, kept_ts, kept_files = [], [], []
    for i in range(n):
        if not os.path.exists(files[i]): continue
        docs.append({"image": str(Path(files[i]).resolve())})
        kept_ts.append(int(ts[i]))
        kept_files.append(Path(files[i]).name)
    return docs, kept_ts, kept_files

class QwenEmbeddingWorker:
    def __init__(self, args, gpu_id: int):
        self.args = args
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        with torch.cuda.device(self.gpu_id):
            self.model = self._load_model()

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        print(f"[GPU {self.gpu_id}] Loading model...", flush=True)
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "device_map": {"": self.gpu_id},
            "attn_implementation": "sdpa"
        }
        if self.args.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
            )
        elif self.args.load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
            
        return SentenceTransformer(self.args.model_id, cache_folder=self.args.cache_dir, trust_remote_code=True, 
                                   model_kwargs=model_kwargs, truncate_dim=self.args.truncate_dim, device=self.device)

    def embed_shard(self, shard_dirs, gpu_shard_idx, checkpoint_dir: Path):
        with torch.cuda.device(self.gpu_id):
            nframes = failed = skipped = 0
            progress = tqdm(shard_dirs, desc=f"GPU {self.gpu_id}", unit="video", position=gpu_shard_idx, leave=True)
            
            for vdir in progress:
                video_id = vdir.name
                ckpt_path = checkpoint_dir / f"{video_id}.pt"
                
                # Tính năng Resume: Bỏ qua nếu đã có checkpoint
                if ckpt_path.exists():
                    skipped += 1
                    continue

                try:
                    docs, ts, frame_files = load_frame_docs(vdir)
                    if not docs: continue
                    
                    emb = self.model.encode(docs, batch_size=self.args.batch_size, normalize_embeddings=True, 
                                           show_progress_bar=False, convert_to_numpy=True).astype(np.float16)
                    
                    # Lưu checkpoint cho TỪNG video ngay lập tức
                    torch.save({
                        "emb": torch.from_numpy(emb),
                        "ts_ms": torch.from_numpy(np.asarray(ts, dtype=np.int32)),
                        "video_ids": [video_id] * len(ts),
                        "frame_files": frame_files
                    }, ckpt_path)
                    
                    nframes += len(ts)
                except Exception as e:
                    failed += 1
                    print(f"\n[GPU {self.gpu_id}] Error {video_id}: {e}")
                
                progress.set_postfix_str(f"new_f={nframes} fail={failed} skip={skipped}")
            
            return nframes

def cmd_embed_pt(args):
    vdirs = list_keyframe_dirs(args.keyframes)
    
    # 1. Tính năng Chia phần (Global Sharding)
    # Ví dụ: --shard-count 2 --shard-index 0 sẽ chạy nửa đầu
    total_videos = len(vdirs)
    chunk_size = total_videos // args.shard_count
    start_idx = args.shard_index * chunk_size
    end_idx = total_videos if args.shard_index == args.shard_count - 1 else (args.shard_index + 1) * chunk_size
    
    active_vdirs = vdirs[start_idx:end_idx]
    if args.limit: active_vdirs = active_vdirs[:args.limit]

    # Tạo thư mục checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== GLOBAL SHARD {args.shard_index+1}/{args.shard_count} ===")
    print(f"Total videos in this shard: {len(active_vdirs)}")
    
    gpu_count = torch.cuda.device_count()
    gpu_shards = [active_vdirs[i::gpu_count] for i in range(gpu_count)]
    
    def run_worker(gpu_id):
        worker = QwenEmbeddingWorker(args, gpu_id)
        return worker.embed_shard(gpu_shards[gpu_id], gpu_id, checkpoint_dir)

    with ThreadPoolExecutor(max_workers=gpu_count) as executor:
        list(executor.map(run_worker, range(gpu_count)))
    
    print(f"\nShard {args.shard_index} processing finished. Checkpoints saved in {checkpoint_dir}")

def cmd_merge(args):
    # Cải tiến lệnh merge: Quét tất cả file .pt trong thư mục checkpoint
    print("Collecting checkpoints...")
    ckpt_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")))
    print(f"Found {len(ckpt_files)} video checkpoints.")
    
    all_emb, all_ts, all_vids, all_frames = [], [], [], []
    for f in tqdm(ckpt_files, desc="Merging"):
        d = torch.load(f, map_location="cpu")
        all_emb.append(d["emb"])
        all_ts.append(d["ts_ms"])
        all_vids.extend(d["video_ids"])
        all_frames.extend(d["frame_files"])
        
    print("Concatenating tensors (this may take a while)...")
    torch.save({
        "emb": torch.cat(all_emb, 0),
        "ts_ms": torch.cat(all_ts, 0),
        "video_ids": all_vids,
        "frame_files": all_frames
    }, args.out)
    print(f"Final index saved to {args.out}")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    
    eb = sub.add_parser("embed-pt")
    eb.add_argument("--model-id", default=MODEL_ID)
    eb.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    eb.add_argument("--keyframes", required=True)
    eb.add_argument("--checkpoint-dir", default="/root/checkpoints")
    eb.add_argument("--out", default="/kaggle/working/final_shard.pt") # Dummy for compatibility
    eb.add_argument("--truncate-dim", type=int, default=1024)
    eb.add_argument("--batch-size", type=int, default=16)
    eb.add_argument("--load-in-8bit", action="store_true")
    eb.add_argument("--load-in-4bit", action="store_true")
    eb.add_argument("--limit", type=int, default=0)
    # Tham số mới để chạy 1/2, 1/3... dữ liệu
    eb.add_argument("--shard-index", type=int, default=0, help="0 to shard-count - 1")
    eb.add_argument("--shard-count", type=int, default=1, help="Total number of parts to split dataset")
    eb.set_defaults(func=cmd_embed_pt)
    
    mg = sub.add_parser("merge")
    mg.add_argument("--checkpoint-dir", default="/root/checkpoints")
    mg.add_argument("--out", required=True)
    mg.set_defaults(func=cmd_merge)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

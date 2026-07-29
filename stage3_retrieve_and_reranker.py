
import argparse
import glob
import json
import os
import gc
import torch
import numpy as np
import traceback
import threading
from pathlib import Path
from typing import Any, List, Dict
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- Cấu hình mặc định ---
DEFAULT_CACHE_DIR = "/root/hf_cache"

# ============================================================
# 1. Utilities (Hỗ trợ load dữ liệu)
# ============================================================
def list_keyframe_dirs(kf_root: str) -> list[Path]:
    return [Path(p).parent for p in sorted(glob.glob(os.path.join(kf_root, "*", "ts_ms.npy")))]

def load_frame_docs(vdir: Path) -> tuple[list[dict[str, str]], list[int], list[str]]:
    files = sorted(glob.glob(str(vdir / "k_*.jpg")))
    ts = np.load(vdir / "ts_ms.npy")
    n = min(len(files), len(ts))
    docs, kept_ts, kept_files = [], [], []
    for i in range(n):
        if not os.path.exists(files[i]): continue
        docs.append({"image": os.path.abspath(files[i])})
        kept_ts.append(int(ts[i]))
        kept_files.append(Path(files[i]).name)
    return docs, kept_ts, kept_files


class IncrementalPredictionWriter:
    """Thread-safe, atomic JSON writer used by multi-GPU reranking.

    The output is rewritten after every completed query. A temporary file is
    fsynced and atomically moved into place, so interruption should leave either
    the previous valid JSON or the newly completed JSON, never a half-written
    output.
    """

    def __init__(
        self,
        output_path: str,
        ordered_task_ids: list[Any],
        overwrite: bool = False,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path = self.output_path.with_name(
            self.output_path.name + ".tmp"
        )
        self.ordered_task_ids = list(ordered_task_ids)
        self.lock = threading.Lock()
        self.results_by_task: dict[Any, dict[str, Any]] = {}

        if overwrite:
            self.output_path.unlink(missing_ok=True)
            self.temp_path.unlink(missing_ok=True)
        elif self.output_path.is_file():
            with open(self.output_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            predictions = payload.get("predictions", [])
            if not isinstance(predictions, list):
                raise ValueError(
                    f"Invalid rerank output: 'predictions' must be a list: "
                    f"{self.output_path}"
                )

            valid_task_ids = set(self.ordered_task_ids)
            for item in predictions:
                if not isinstance(item, dict) or "task_id" not in item:
                    continue
                task_id = item["task_id"]
                if task_id in valid_task_ids:
                    self.results_by_task[task_id] = item

            print(
                f"[Resume] Loaded {len(self.results_by_task)}/"
                f"{len(self.ordered_task_ids)} completed queries from "
                f"{self.output_path}",
                flush=True,
            )

    def completed_task_ids(self) -> set[Any]:
        with self.lock:
            return set(self.results_by_task)

    def save(self, prediction: dict[str, Any]) -> int:
        """Save one completed query and return total completed-query count."""
        task_id = prediction["task_id"]

        with self.lock:
            self.results_by_task[task_id] = prediction
            ordered_predictions = [
                self.results_by_task[current_task_id]
                for current_task_id in self.ordered_task_ids
                if current_task_id in self.results_by_task
            ]

            with open(self.temp_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {"predictions": ordered_predictions},
                    handle,
                    ensure_ascii=False,
                )
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(self.temp_path, self.output_path)
            return len(ordered_predictions)


# ============================================================
# 2. Worker Embedding (Đa GPU - Stage 1)
# ============================================================
class QwenEmbeddingWorker:
    def __init__(self, args, gpu_id: int):
        self.args = args
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        with torch.cuda.device(self.gpu_id):
            self.model = self._load_model()

    def _load_model(self):
        print(f"[GPU {self.gpu_id}] Loading Embedding Model: {self.args.model_id}", flush=True)
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
            
        return SentenceTransformer(self.args.model_id, cache_folder=self.args.cache_dir, 
                                   trust_remote_code=True, model_kwargs=model_kwargs, 
                                   truncate_dim=self.args.truncate_dim, device=self.device)

    def run(self, shard_dirs, gpu_shard_idx, checkpoint_dir: Path):
        with torch.cuda.device(self.gpu_id):
            nframes = failed = skipped = 0
            progress = tqdm(shard_dirs, desc=f"GPU {self.gpu_id}", position=gpu_shard_idx, leave=True)
            
            for vdir in progress:
                video_id = vdir.name
                ckpt_path = checkpoint_dir / f"{video_id}.pt"
                if ckpt_path.exists():
                    skipped += 1
                    continue
                try:
                    docs, ts, frame_files = load_frame_docs(vdir)
                    if not docs: continue
                    emb = self.model.encode(docs, batch_size=self.args.batch_size, 
                                           normalize_embeddings=True, show_progress_bar=False, 
                                           convert_to_numpy=True).astype(np.float16)
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
                progress.set_postfix_str(f"f={nframes} fail={failed} skip={skipped}")
            return nframes

# ============================================================
# 3. Retrieval Engine (Stage 1.5)
# ============================================================
class QwenRetrievalEngine:
    def __init__(self, args):
        self.args = args
        self.device = "cuda:0"

    def _load_model(self):
        print(f"[Retrieval] Loading Embedding Model for Queries: {self.args.model_id}...")
        model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16, "device_map": {"": self.device}, "attn_implementation": "sdpa"}
        if self.args.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        return SentenceTransformer(self.args.model_id, cache_folder=self.args.cache_dir, trust_remote_code=True, model_kwargs=model_kwargs, truncate_dim=self.args.truncate_dim, device=self.device)

    def run(self):
        print(f"[Retrieval] Loading index: {self.args.index_pt}...")
        idx = torch.load(self.args.index_pt, map_location="cpu")
        emb_gallery = idx["emb"].to(self.device).half()
        ts_ms, video_ids = idx["ts_ms"].numpy(), idx["video_ids"]
        
        model = self._load_model()
        tasks = [json.loads(line) for line in open(self.args.tasks, "r") if line.strip()]
        queries = [t["description"] for t in tasks]
        q_emb = model.encode(queries, convert_to_tensor=True, show_progress_bar=True).half()
        
        del model
        torch.cuda.empty_cache()

        print(f"[Retrieval] Scoring {emb_gallery.shape[0]} frames...")
        cand_k = min(self.args.cand_keyframes, emb_gallery.shape[0])
        predictions = []
        for i in range(len(tasks)):
            sims = torch.matmul(q_emb[i], emb_gallery.T)
            vals, idxs = torch.topk(sims, k=cand_k)
            vals, idxs = vals.cpu().float().numpy(), idxs.cpu().numpy()
            
            selected_v, per_v_counts, results = set(), {}, []
            for score, idx_f in zip(vals, idxs):
                vid = str(video_ids[idx_f])
                if vid not in per_v_counts:
                    if len(selected_v) >= self.args.top_videos: continue
                    selected_v.add(vid); per_v_counts[vid] = 0
                if per_v_counts[vid] >= self.args.frames_per_video: continue
                results.append({"video_id": vid, "frame_ms": int(ts_ms[idx_f]), "embed_score": float(score)})
                per_v_counts[vid] += 1
                if len(selected_v) >= self.args.top_videos and all(c >= self.args.frames_per_video for c in per_v_counts.values()): break
            predictions.append({"task_id": tasks[i]["task_id"], "description": tasks[i].get("description",""), "results": results})
            
        with open(self.args.out, "w") as f:
            json.dump({"predictions": predictions}, f, ensure_ascii=False)
        print(f"[Done] Retrieval results saved to {self.args.out}")

# ============================================================
# 4. Worker Reranker (Đa GPU - Stage 2)
# ============================================================
class QwenRerankWorker:
    def __init__(self, args, gpu_id: int):
        self.args = args
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        with torch.cuda.device(self.gpu_id):
            self.model = self._load_model()

    def _load_model(self):
        print(f"[GPU {self.gpu_id}] Loading Reranker: {self.args.model_id}...", flush=True)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        return CrossEncoder(self.args.model_id, model_kwargs={"trust_remote_code": True, "quantization_config": bnb_config, "device_map": {"": self.gpu_id}, "attn_implementation": "sdpa"})

    def run(
        self,
        tasks_shard,
        lookup,
        gpu_shard_idx,
        writer: IncrementalPredictionWriter,
    ):
        completed_here = 0
        kf_path = Path(self.args.keyframes)
        pbar = tqdm(
            tasks_shard,
            desc=f"GPU {self.gpu_id} Reranking",
            position=gpu_shard_idx,
            leave=True,
        )

        for local_index, task in enumerate(pbar, start=1):
            task_id = task["task_id"]
            query = task.get("description", "")
            candidates = task["results"][:self.args.rerank_top]
            pairs, valid_cands = [], []

            for cand in candidates:
                img_file = lookup.get(
                    (str(cand["video_id"]), int(cand["frame_ms"]))
                )
                if not img_file:
                    continue

                img_p = kf_path / cand["video_id"] / img_file
                if img_p.exists():
                    pairs.append((query, str(img_p)))
                    # Copy the candidate so rerank_score does not mutate the
                    # shared candidates object used by another thread.
                    valid_cands.append(dict(cand))

            if pairs:
                with torch.cuda.device(self.gpu_id):
                    scores = self.model.predict(
                        pairs,
                        batch_size=1,
                        show_progress_bar=False,
                    )

                    # Force asynchronous CUDA failures to surface on the query
                    # that caused them, before that query is marked completed.
                    torch.cuda.synchronize(self.gpu_id)

                for cand, score in zip(valid_cands, scores):
                    cand["rerank_score"] = float(score)

                valid_cands.sort(
                    key=lambda item: item.get("rerank_score", -99),
                    reverse=True,
                )

            result = [
                {
                    "rank": rank,
                    "video_id": candidate["video_id"],
                    "frame_ms": candidate["frame_ms"],
                }
                for rank, candidate in enumerate(
                    valid_cands[:self.args.final_top],
                    start=1,
                )
            ]

            prediction = {
                "task_id": task_id,
                "results": result,
            }

            # Atomic, thread-safe save immediately after this query succeeds.
            total_saved = writer.save(prediction)
            completed_here += 1
            pbar.set_postfix_str(
                f"saved_total={total_saved} this_gpu={completed_here}"
            )

            # Avoid empty_cache() after every query. It does not free the model
            # and can make repeated allocation slower. Run it only periodically.
            if local_index % 25 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        return completed_here

# ============================================================
# 5. Main Orchestrator (Điều phối lệnh)
# ============================================================
def cmd_embed_pt(args):
    vdirs = list_keyframe_dirs(args.keyframes)
    chunk_size = len(vdirs) // args.shard_count
    vdirs = vdirs[args.shard_index*chunk_size : (len(vdirs) if args.shard_index==args.shard_count-1 else (args.shard_index+1)*chunk_size)]
    checkpoint_dir = Path(args.checkpoint_dir); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    gpu_count = torch.cuda.device_count()
    gpu_shards = [vdirs[i::gpu_count] for i in range(gpu_count)]
    print(f"=== EMBEDDING SHARD {args.shard_index+1}/{args.shard_count} | {gpu_count} GPUs ===")
    def worker_thread(gpu_id):
        worker = QwenEmbeddingWorker(args, gpu_id)
        return worker.run(gpu_shards[gpu_id], gpu_id, checkpoint_dir)
    with ThreadPoolExecutor(max_workers=gpu_count) as executor:
        list(executor.map(worker_thread, range(gpu_count)))

def cmd_merge(args):
    ckpt_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")))
    print(f"Merging {len(ckpt_files)} files...")
    all_emb, all_ts, all_vids, all_frames = [], [], [], []
    for f in tqdm(ckpt_files):
        d = torch.load(f, map_location="cpu")
        all_emb.append(d["emb"]); all_ts.append(d["ts_ms"]); all_vids.extend(d["video_ids"]); all_frames.extend(d["frame_files"])
    torch.save({"emb": torch.cat(all_emb, 0), "ts_ms": torch.cat(all_ts, 0), "video_ids": all_vids, "frame_files": all_frames}, args.out)
    print(f"Index saved: {args.out}")

def cmd_rerank(args):
    with open(args.candidates, "r", encoding="utf-8") as handle:
        candidates = json.load(handle)["predictions"]

    with open(args.tasks, "r", encoding="utf-8") as handle:
        ordered_ids = [
            json.loads(line)["task_id"]
            for line in handle
            if line.strip()
        ]

    writer = IncrementalPredictionWriter(
        output_path=args.out,
        ordered_task_ids=ordered_ids,
        overwrite=args.overwrite_output,
    )

    completed_ids = writer.completed_task_ids()
    pending_candidates = [
        task
        for task in candidates
        if task["task_id"] not in completed_ids
    ]

    print(
        f"[Main] Queries: total={len(ordered_ids)} | "
        f"completed={len(completed_ids)} | "
        f"pending={len(pending_candidates)}",
        flush=True,
    )

    if not pending_candidates:
        print(f"[Done] Nothing to rerank. Output: {args.out}", flush=True)
        return

    print(f"[Main] Building lookup from {args.index_pt}...")
    idx = torch.load(args.index_pt, map_location="cpu")
    lookup = {
        (str(video_id), int(timestamp)): frame_file
        for video_id, timestamp, frame_file in zip(
            idx["video_ids"],
            idx["ts_ms"],
            idx["frame_files"],
        )
    }
    del idx
    gc.collect()

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        raise RuntimeError("No CUDA GPU is available for reranking")

    shards = [
        pending_candidates[gpu_id::gpu_count]
        for gpu_id in range(gpu_count)
    ]

    def run_worker(gpu_id):
        if not shards[gpu_id]:
            return 0

        worker = QwenRerankWorker(args, gpu_id)
        return worker.run(
            shards[gpu_id],
            lookup,
            gpu_id,
            writer,
        )

    try:
        with ThreadPoolExecutor(max_workers=gpu_count) as executor:
            completed_per_gpu = list(
                executor.map(run_worker, range(gpu_count))
            )
    except Exception:
        saved_count = len(writer.completed_task_ids())
        print(
            f"\n[Interrupted] Reranking stopped, but "
            f"{saved_count}/{len(ordered_ids)} completed queries are safely "
            f"stored in {args.out}",
            flush=True,
        )
        raise

    total_saved = len(writer.completed_task_ids())
    print(
        f"[Done] Submission saved incrementally to {args.out} | "
        f"completed={total_saved}/{len(ordered_ids)} | "
        f"new_per_gpu={completed_per_gpu}",
        flush=True,
    )

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    
    eb = sub.add_parser("embed-pt")
    eb.add_argument("--model-id", required=True); eb.add_argument("--keyframes", required=True)
    eb.add_argument("--checkpoint-dir", default="/kaggle/working/checkpoints")
    eb.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR); eb.add_argument("--truncate-dim", type=int, default=1024)
    eb.add_argument("--batch-size", type=int, default=32); eb.add_argument("--load-in-4bit", action="store_true")
    eb.add_argument("--load-in-8bit", action="store_true"); eb.add_argument("--shard-index", type=int, default=0); eb.add_argument("--shard-count", type=int, default=1)
    eb.set_defaults(func=cmd_embed_pt)
    
    rt = sub.add_parser("retrieve")
    rt.add_argument("--model-id", required=True); rt.add_argument("--index-pt", required=True); rt.add_argument("--tasks", required=True)
    rt.add_argument("--out", default="candidates.json"); rt.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR); rt.add_argument("--truncate-dim", type=int, default=1024)
    rt.add_argument("--top-videos", type=int, default=5006); rt.add_argument("--frames-per-video", type=int, default=257); rt.add_argument("--cand-keyframes", type=int, default=300000)
    rt.add_argument("--load-in-4bit", action="store_true"); rt.set_defaults(func=lambda args: QwenRetrievalEngine(args).run())
    
    mg = sub.add_parser("merge")
    mg.add_argument("--checkpoint-dir", required=True); mg.add_argument("--out", required=True)
    mg.set_defaults(func=cmd_merge)
    
    rr = sub.add_parser("rerank")
    rr.add_argument("--model-id", required=True); rr.add_argument("--candidates", required=True); rr.add_argument("--tasks", required=True)
    rr.add_argument("--index-pt", required=True); rr.add_argument("--keyframes", required=True); rr.add_argument("--out", default="submission.json")
    rr.add_argument("--rerank-top", type=int, default=50); rr.add_argument("--final-top", type=int, default=10); rr.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    rr.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Ignore/delete an existing partial rerank output and start again.",
    )
    rr.set_defaults(func=cmd_rerank)
    
    args = parser.parse_args(); args.func(args)

if __name__ == "__main__":
    main()

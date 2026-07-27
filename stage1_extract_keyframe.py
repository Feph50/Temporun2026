"""Stage 1 replacement using OmniShotCut shot ranges.

Output is compatible with TempoRun2026_Baseline:
  <out>/<video_id>/k_00001.jpg, k_00002.jpg, ...
  <out>/<video_id>/ts_ms.npy

Each saved frame is the midpoint of one OmniShotCut-detected shot.  The timestamp
is clip-relative milliseconds, aligned with the sorted JPG names.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download


REPO_ROOT = "OmniShotCut"
'''if not REPO_ROOT.exists():
    raise SystemExit(f"OmniShotCut repo not found at {REPO_ROOT}")'''
sys.path.insert(0, str(REPO_ROOT))

from omnishotcut.architecture.backbone import build_backbone
from omnishotcut.architecture.model import OmniShotCut
from omnishotcut.architecture.transformer import build_transformer
from omnishotcut.engine import merge_predictions, split_videos, video_transform
from omnishotcut.label_correspondence import unique_intra_label_mapping


HF_REPO = "uva-cv-lab/OmniShotCut"
HF_FILENAME = "OmniShotCut_ckpt.pth"


def list_videos(roots: list[str]) -> list[tuple[str, str]]:
    videos: list[tuple[str, str]] = []
    for root in roots:
        coll = Path(root).name.lower()
        for mp4 in sorted(glob.glob(os.path.join(root, "videos", "*", "*.mp4"))):
            videos.append((f"{coll}_{Path(mp4).stem}", mp4))
    return videos


def load_model_device(checkpoint_path: str | None, device: str):
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = state["args"]
    model = OmniShotCut(
        build_backbone(model_args),
        build_transformer(model_args),
        num_intra_relation_classes=model_args.num_intra_relation_classes,
        num_inter_relation_classes=model_args.num_inter_relation_classes,
        num_frames=model_args.max_process_window_length,
        num_queries=model_args.num_queries,
        aux_loss=model_args.aux_loss,
    )
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()
    return model, model_args


def infer_ranges(video_rgb: np.ndarray, model, model_args, device: str, overlap: int, mode: str):
    pred_boundary_full = []
    for video_chunk, _pad, window_start, valid_start, valid_end, valid_len in split_videos(
        video_rgb, model_args.max_process_window_length, overlap
    ):
        video_tensor = video_transform(video_chunk).unsqueeze(0).to(device)
        with torch.inference_mode():
            outputs = model(video_tensor)

        probas_intra = outputs["intra_clip_logits"].softmax(-1)[0, :, :-1]
        probas_inter = outputs["inter_clip_logits"].softmax(-1)[0, :, :-1]
        range_probas = outputs["pred_shot_logits"].softmax(-1)[0, :, :-1]
        query_intra_idx = probas_intra.argmax(dim=-1)
        query_inter_idx = probas_inter.argmax(dim=-1)
        query_range_idx = range_probas.argmax(dim=-1)

        pred_boundary = []
        start_local = 0
        for keep_idx in range(len(query_intra_idx)):
            intra_label = int(query_intra_idx[keep_idx].detach().cpu())
            inter_label = int(query_inter_idx[keep_idx].detach().cpu())
            end_local = int(query_range_idx[keep_idx].detach().cpu())
            end_local = min(end_local, valid_len)
            if start_local >= end_local:
                continue

            end_global = window_start + end_local
            if valid_start < end_global <= valid_end:
                pred_boundary.append(
                    {
                        "end_frame_idx": int(end_global),
                        "intra_label": intra_label,
                        "inter_label": inter_label,
                    }
                )
            start_local = end_local
            if end_local >= valid_len:
                break

        pred_boundary_full = merge_predictions(pred_boundary_full, pred_boundary)

    ranges = []
    labels = []
    start = 0
    for item in pred_boundary_full:
        end = int(item["end_frame_idx"])
        if end <= start:
            continue
        ranges.append([start, end])
        labels.append(int(item["intra_label"]))
        start = end

    if not ranges or ranges[-1][1] < len(video_rgb) - 1:
        ranges.append([ranges[-1][1] if ranges else 0, len(video_rgb) - 1])
        labels.append(unique_intra_label_mapping["general"])

    if mode == "clean_shot":
        general = unique_intra_label_mapping["general"]
        ranges = [r for r, label in zip(ranges, labels) if label == general]
    return ranges


def read_video_for_model(mp4: str, width: int, height: int) -> tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {mp4}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError("no readable frames")
    return np.asarray(frames, dtype=np.uint8), float(fps), total or len(frames)


def save_midpoint_frames(mp4: str, out_dir: Path, ranges: list[list[int]], fps: float, total_frames: int) -> list[int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for extraction: {mp4}")

    ts_ms = []
    for idx, (start, end) in enumerate(ranges, 1):
        frame_idx = max(0, min(total_frames - 1, int(round((start + end) / 2))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(out_dir / f"k_{idx:05d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        ts_ms.append(int(round(frame_idx * 1000.0 / fps)))
    cap.release()
    if not ts_ms:
        raise RuntimeError("no midpoint frames saved")
    np.save(out_dir / "ts_ms.npy", np.asarray(ts_ms, dtype=np.int32))
    return ts_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["default", "clean_shot"], default="default")
    parser.add_argument("--overlap-window-length", type=int, default=20)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fail_log = out / f"failed_omnishotcut_shard{args.shard_index}.txt"

    videos = list_videos(args.dataset_root)
    if not videos:
        raise SystemExit("no videos found")
    mine = [item for i, item in enumerate(videos) if i % args.shard_count == args.shard_index]
    if args.limit:
        mine = mine[: args.limit]

    print(f"[device] {args.device} | torch={torch.__version__} | cuda={torch.cuda.is_available()}", flush=True)
    print(f"[shard {args.shard_index}/{args.shard_count}] {len(mine)}/{len(videos)} videos", flush=True)
    model, model_args = load_model_device(args.checkpoint_path, args.device)
    width, height = int(model_args.process_width), int(model_args.process_height)

    started = time.time()
    done = saved = failed = 0
    for video_id, mp4 in mine:
        vdir = out / video_id
        ts_path = vdir / "ts_ms.npy"
        if ts_path.exists():
            done += 1
            continue
        tmp_dir = out / f".{video_id}.tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            video_rgb, fps, total_frames = read_video_for_model(mp4, width, height)
            ranges = infer_ranges(video_rgb, model, model_args, args.device, args.overlap_window_length, args.mode)
            ts_ms = save_midpoint_frames(mp4, tmp_dir, ranges, fps, total_frames)
            with open(tmp_dir / "omnishotcut_ranges.json", "w", encoding="utf-8") as f:
                json.dump({"video_id": video_id, "mp4": mp4, "fps": fps, "ranges": ranges}, f)
            if vdir.exists():
                shutil.rmtree(vdir)
            tmp_dir.rename(vdir)
            saved += len(ts_ms)
        except Exception as exc:
            failed += 1
            shutil.rmtree(tmp_dir, ignore_errors=True)
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"{video_id}\t{mp4}\t{type(exc).__name__}: {exc}\n")
        done += 1
        if done % 10 == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[shard {args.shard_index}] {done}/{len(mine)} videos | {saved} frames | "
                f"fail={failed} | {done / elapsed * 60:.1f} vids/min",
                flush=True,
            )

    print(f"[done] videos={done} frames={saved} failed={failed} out={out}", flush=True)


if __name__ == "__main__":
    main()

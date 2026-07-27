"""Expand existing OmniShotCut midpoint keyframes without re-running shot detection.

Input layout:
  <ranges-root>/<video_id>/omnishotcut_ranges.json
  <ranges-root>/<video_id>/ts_ms.npy
  <ranges-root>/<video_id>/k_00001.jpg, k_00002.jpg, ...

Default output contains only newly selected EXTRA frames:
  <out>/<video_id>/k_000001.jpg, k_000002.jpg, ...
  <out>/<video_id>/ts_ms.npy
  <out>/<video_id>/omnishotcut_ranges.json
  <out>/<video_id>/keyframe_selection.json

Existing midpoint JPGs are still used internally as ResNet comparison anchors,
but they are not copied to the new output when --output-mode=extras_only.
Use --output-mode=all only when a standalone combined middle+extra folder is
needed.

Selection policy:
  <5s      : keep only the existing midpoint
  5-<10s   : midpoint + at most 1 extra frame
  10-<30s  : midpoint + at most 2 extra frames
  >=30s    : midpoint + about 1 extra per 15 seconds, max 10 total

Extra-frame candidates are sampled at 1 FPS by default. Their embeddings are
computed using the ResNet backbone already stored in the OmniShotCut checkpoint.
A candidate is skipped when its cosine similarity to any selected frame is
greater than the configured threshold.

Shot ranges and midpoint timestamps are read from the existing JSON/NPY files,
so this script does not run OmniShotCut shot-boundary inference again.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parent / "OmniShotCut"
if not REPO_ROOT.exists():
    raise SystemExit(f"OmniShotCut repo not found at {REPO_ROOT}")
sys.path.insert(0, str(REPO_ROOT))

# Prevent build_backbone() from downloading torchvision ImageNet weights.
# The complete fine-tuned backbone weights are loaded from OmniShotCut_ckpt.pth.
import omnishotcut.architecture.backbone as backbone_module
from omnishotcut.engine import video_transform

backbone_module.is_main_process = lambda: False
build_backbone = backbone_module.build_backbone


HF_REPO = "uva-cv-lab/OmniShotCut"
HF_FILENAME = "OmniShotCut_ckpt.pth"
DONE_FILENAME = "keyframe_selection.json"


def list_videos(roots: list[str]) -> dict[str, str]:
    """Fallback lookup used only when the MP4 path saved in JSON no longer exists."""
    videos: dict[str, str] = {}
    for root in roots:
        coll = Path(root).name.lower()
        for mp4 in sorted(glob.glob(os.path.join(root, "videos", "*", "*.mp4"))):
            videos[f"{coll}_{Path(mp4).stem}"] = mp4
    return videos


def list_range_dirs(ranges_root: Path, video_id_regex: str) -> list[Path]:
    pattern = re.compile(video_id_regex, flags=re.IGNORECASE)
    result = []
    for path in sorted(ranges_root.iterdir()):
        if not path.is_dir() or pattern.fullmatch(path.name) is None:
            continue
        if (path / "omnishotcut_ranges.json").is_file() and (path / "ts_ms.npy").is_file():
            result.append(path)
    return result


def load_backbone_device(
    checkpoint_path: str | None,
    device: str,
    local_files_only: bool,
):
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILENAME,
            local_files_only=local_files_only,
        )
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "args" not in state or "model" not in state:
        raise ValueError("checkpoint must contain keys 'args' and 'model'")

    model_args = state["args"]
    backbone = build_backbone(model_args)

    prefix = "backbone."
    backbone_state = {
        key[len(prefix) :]: value
        for key, value in state["model"].items()
        if key.startswith(prefix)
    }
    if not backbone_state:
        raise RuntimeError("no 'backbone.*' parameters found in OmniShotCut checkpoint")

    backbone.load_state_dict(backbone_state, strict=True)
    backbone_body = backbone[0].body
    backbone_body.to(device)
    backbone_body.eval()

    return backbone_body, model_args, checkpoint_path


def shot_policy(duration_sec: float) -> tuple[int, float, str]:
    """Return total-frame budget, minimum temporal gap, and policy name."""
    if duration_sec < 5.0:
        return 1, 0.0, "middle_only"
    if duration_sec < 10.0:
        return 2, 1.0, "middle_plus_1"
    if duration_sec < 30.0:
        return 3, 2.0, "middle_plus_2"

    # One mandatory midpoint + approximately one extra frame per 15 seconds.
    total_budget = min(10, 1 + int(math.ceil(duration_sec / 15.0)))
    return total_budget, 3.0, "long_dynamic"


def midpoint_from_range(start: int, end: int) -> int:
    return int(round((start + end) / 2.0))


def candidate_indices(
    start: int,
    end: int,
    fps: float,
    candidate_fps: float,
    middle_frame: int,
) -> list[int]:
    """Sample candidates near the center of each sampling interval, avoiding cut boundaries."""
    if end <= start or candidate_fps <= 0:
        return []

    stride = max(1, int(round(fps / candidate_fps)))
    first = start + stride // 2
    indices = list(range(first, end, stride))

    # Do not embed the same exact frame twice.
    return [idx for idx in indices if idx != middle_frame]


def read_frame_hybrid(
    cap: cv2.VideoCapture,
    frame_idx: int,
    state: dict[str, int | None],
    max_decode_gap_frames: int,
) -> np.ndarray | None:
    """Read sorted frame indices using short sequential decodes and sparse seeks."""
    next_idx = state.get("next_idx")

    if (
        next_idx is not None
        and frame_idx >= next_idx
        and frame_idx - next_idx <= max_decode_gap_frames
    ):
        frame = None
        while next_idx <= frame_idx:
            ok, current = cap.read()
            if not ok:
                state["next_idx"] = None
                return None
            if next_idx == frame_idx:
                frame = current
            next_idx += 1
        state["next_idx"] = next_idx
        return frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    state["next_idx"] = frame_idx + 1 if ok else None
    return frame if ok else None


def embed_rgb_batch(
    frames_rgb: list[np.ndarray],
    backbone_body,
    device: str,
    use_amp: bool,
) -> np.ndarray:
    if not frames_rgb:
        return np.empty((0, 0), dtype=np.float32)

    video_np = np.stack(frames_rgb, axis=0).astype(np.uint8, copy=False)
    tensor = video_transform(video_np).to(device, non_blocking=True)

    amp_enabled = use_amp and str(device).startswith("cuda")
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            feature_dict = backbone_body(tensor)
            feature_map = list(feature_dict.values())[-1]
            embeddings = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)

        embeddings = F.normalize(embeddings.float(), p=2, dim=1)

    return embeddings.cpu().numpy()


def embed_image_paths(
    path_by_frame: dict[int, Path],
    width: int,
    height: int,
    backbone_body,
    device: str,
    batch_size: int,
    use_amp: bool,
) -> tuple[dict[int, np.ndarray], list[int]]:
    embeddings: dict[int, np.ndarray] = {}
    failed: list[int] = []
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []

    def flush() -> None:
        nonlocal batch_frames, batch_indices
        if not batch_frames:
            return
        values = embed_rgb_batch(batch_frames, backbone_body, device, use_amp)
        for idx, embedding in zip(batch_indices, values):
            embeddings[idx] = embedding
        batch_frames = []
        batch_indices = []

    for frame_idx, path in sorted(path_by_frame.items()):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            failed.append(frame_idx)
            continue
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_frames.append(frame_rgb)
        batch_indices.append(frame_idx)
        if len(batch_frames) >= batch_size:
            flush()

    flush()
    return embeddings, failed


def embed_video_indices(
    mp4: str,
    indices: Iterable[int],
    width: int,
    height: int,
    fps: float,
    backbone_body,
    device: str,
    batch_size: int,
    use_amp: bool,
    sequential_gap_sec: float,
) -> tuple[dict[int, np.ndarray], list[int]]:
    indices = sorted(set(int(idx) for idx in indices))
    embeddings: dict[int, np.ndarray] = {}
    failed: list[int] = []
    if not indices:
        return embeddings, failed

    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for candidate embedding: {mp4}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_decode_gap_frames = max(0, int(round(fps * sequential_gap_sec)))
    read_state: dict[str, int | None] = {"next_idx": None}
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []

    def flush() -> None:
        nonlocal batch_frames, batch_indices
        if not batch_frames:
            return
        values = embed_rgb_batch(batch_frames, backbone_body, device, use_amp)
        for idx, embedding in zip(batch_indices, values):
            embeddings[idx] = embedding
        batch_frames = []
        batch_indices = []

    try:
        for frame_idx in indices:
            if total_frames > 0 and not 0 <= frame_idx < total_frames:
                failed.append(frame_idx)
                continue

            frame = read_frame_hybrid(
                cap,
                frame_idx,
                read_state,
                max_decode_gap_frames=max_decode_gap_frames,
            )
            if frame is None:
                failed.append(frame_idx)
                continue

            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame_rgb)
            batch_indices.append(frame_idx)

            if len(batch_frames) >= batch_size:
                flush()

        flush()
    finally:
        cap.release()

    return embeddings, failed


def best_novel_candidate(
    candidates: list[int],
    selected: list[int],
    embeddings: dict[int, np.ndarray],
    min_gap_frames: int,
) -> tuple[int | None, float | None]:
    best_idx = None
    best_max_similarity = float("inf")
    best_temporal_distance = -1

    for candidate in candidates:
        if candidate not in embeddings:
            continue
        if any(abs(candidate - kept) < min_gap_frames for kept in selected):
            continue

        similarities = [
            float(np.dot(embeddings[candidate], embeddings[kept]))
            for kept in selected
            if kept in embeddings
        ]
        if not similarities:
            continue

        max_similarity = max(similarities)
        temporal_distance = min(abs(candidate - kept) for kept in selected)

        if (
            max_similarity < best_max_similarity
            or (
                math.isclose(max_similarity, best_max_similarity, abs_tol=1e-8)
                and temporal_distance > best_temporal_distance
            )
        ):
            best_idx = candidate
            best_max_similarity = max_similarity
            best_temporal_distance = temporal_distance

    if best_idx is None:
        return None, None
    return best_idx, float(best_max_similarity)


def select_shot_frames(
    plan: dict[str, Any],
    embeddings: dict[int, np.ndarray],
    threshold: float,
) -> list[dict[str, Any]]:
    middle = int(plan["middle_frame"])
    extra_budget = int(plan["total_budget"]) - 1
    min_gap_frames = int(round(plan["min_gap_sec"] * plan["fps"]))

    selected = [middle]
    selected_info: list[dict[str, Any]] = [
        {
            "frame_idx": middle,
            "kind": "middle",
            "max_similarity": None,
        }
    ]

    if extra_budget <= 0 or middle not in embeddings:
        return selected_info

    remaining = [
        int(idx)
        for idx in plan["candidate_frames"]
        if int(idx) in embeddings and int(idx) != middle
    ]

    def try_add(pool: list[int]) -> bool:
        nonlocal remaining
        candidate, max_similarity = best_novel_candidate(
            pool,
            selected,
            embeddings,
            min_gap_frames=min_gap_frames,
        )
        if candidate is None or max_similarity is None:
            return False

        # Requested behavior: similarity > threshold means skip.
        if max_similarity > threshold:
            return False

        selected.append(candidate)
        selected_info.append(
            {
                "frame_idx": int(candidate),
                "kind": "extra",
                "max_similarity": round(float(max_similarity), 6),
            }
        )
        remaining = [idx for idx in remaining if idx != candidate]
        return True

    duration_sec = float(plan["duration_sec"])

    # For 10-30 second shots, first try one frame on each side of the midpoint.
    if 10.0 <= duration_sec < 30.0 and extra_budget >= 2:
        left = [idx for idx in remaining if idx < middle]
        right = [idx for idx in remaining if idx > middle]

        if try_add(left):
            extra_budget -= 1
        if extra_budget > 0 and try_add(right):
            extra_budget -= 1

    # Fill any remaining budget with max-min diversity over the whole shot.
    while extra_budget > 0:
        if not try_add(remaining):
            break
        extra_budget -= 1

    return selected_info


def resolve_mp4(
    video_id: str,
    metadata: dict[str, Any],
    fallback_lookup: dict[str, str],
) -> str:
    saved_path = str(metadata.get("mp4", "")).strip()
    if saved_path and os.path.isfile(saved_path):
        return saved_path

    fallback = fallback_lookup.get(video_id)
    if fallback and os.path.isfile(fallback):
        return fallback

    raise FileNotFoundError(
        f"video source not found for {video_id}; JSON path={saved_path!r}"
    )


def existing_middle_mapping(
    source_dir: Path,
    ranges: list[list[int]],
    fps: float,
    ts_ms: np.ndarray,
) -> tuple[list[int], list[int], dict[int, Path], bool]:
    """Return midpoint frame indices/timestamps and reusable JPGs."""
    exact_ts_alignment = len(ts_ms) == len(ranges)
    midpoint_frames: list[int] = []
    midpoint_timestamps: list[int] = []
    jpg_by_shot: dict[int, Path] = {}

    for shot_idx, (start, end) in enumerate(ranges):
        computed_mid = midpoint_from_range(start, end)

        if exact_ts_alignment:
            timestamp = int(ts_ms[shot_idx])
            frame_idx = int(round(timestamp * fps / 1000.0))
            # Keep timestamp-derived midpoint inside the shot range.
            high = max(start, end - 1)
            frame_idx = max(start, min(high, frame_idx))
        else:
            frame_idx = computed_mid
            timestamp = int(round(frame_idx * 1000.0 / fps))

        midpoint_frames.append(frame_idx)
        midpoint_timestamps.append(timestamp)

        expected_jpg = source_dir / f"k_{shot_idx + 1:05d}.jpg"
        if expected_jpg.is_file():
            jpg_by_shot[shot_idx] = expected_jpg

    return midpoint_frames, midpoint_timestamps, jpg_by_shot, exact_ts_alignment


def write_selected_frames(
    mp4: str,
    temp_dir: Path,
    selected_items: list[dict[str, Any]],
    middle_jpg_by_shot: dict[int, Path],
    fps: float,
    jpeg_quality: int,
    sequential_gap_sec: float,
    filename_digits: int,
) -> list[int]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    needs_video = any(
        not (
            item["kind"] == "middle"
            and int(item["shot_index"]) in middle_jpg_by_shot
        )
        for item in selected_items
    )

    cap = None
    read_state: dict[str, int | None] = {"next_idx": None}
    max_decode_gap_frames = max(0, int(round(fps * sequential_gap_sec)))
    cached_idx = None
    cached_frame = None

    if needs_video:
        cap = cv2.VideoCapture(mp4)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video for final extraction: {mp4}")

    timestamps: list[int] = []
    try:
        for output_idx, item in enumerate(selected_items, start=1):
            destination = temp_dir / f"k_{output_idx:0{filename_digits}d}.jpg"
            frame_idx = int(item["frame_idx"])
            shot_idx = int(item["shot_index"])

            source_jpg = (
                middle_jpg_by_shot.get(shot_idx)
                if item["kind"] == "middle"
                else None
            )
            if source_jpg is not None and source_jpg.is_file():
                shutil.copy2(source_jpg, destination)
            else:
                assert cap is not None
                if cached_idx == frame_idx and cached_frame is not None:
                    frame = cached_frame
                else:
                    frame = read_frame_hybrid(
                        cap,
                        frame_idx,
                        read_state,
                        max_decode_gap_frames=max_decode_gap_frames,
                    )
                    if frame is None:
                        raise RuntimeError(f"cannot read selected frame {frame_idx}")
                    cached_idx = frame_idx
                    cached_frame = frame

                ok = cv2.imwrite(
                    str(destination),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if not ok:
                    raise RuntimeError(f"cannot write image: {destination}")

            timestamps.append(int(item["timestamp_ms"]))
    finally:
        if cap is not None:
            cap.release()

    # A video may legitimately have no accepted extra frames.  Keep an empty
    # ts_ms.npy plus keyframe_selection.json so resume logic can still mark the
    # video as completed.
    np.save(temp_dir / "ts_ms.npy", np.asarray(timestamps, dtype=np.int32))
    return timestamps


def same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def commit_temp_dir(
    temp_dir: Path,
    destination: Path,
    source_dir: Path,
    overwrite: bool,
) -> None:
    """Commit output safely, including supported in-place replacement."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if same_path(destination, source_dir):
        backup = destination.parent / f".{destination.name}.middle_backup"
        shutil.rmtree(backup, ignore_errors=True)

        source_dir.rename(backup)
        try:
            temp_dir.rename(destination)
        except Exception:
            if destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            backup.rename(source_dir)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        return

    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"destination already exists without a completed marker: {destination}; "
                "use --overwrite to replace it"
            )
        shutil.rmtree(destination)

    temp_dir.rename(destination)


def process_video(
    source_dir: Path,
    destination_dir: Path,
    fallback_lookup: dict[str, str],
    backbone_body,
    model_args,
    args,
) -> tuple[int, int, int]:
    video_id = source_dir.name
    ranges_path = source_dir / "omnishotcut_ranges.json"
    ts_path = source_dir / "ts_ms.npy"

    with open(ranges_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    raw_ranges = metadata.get("ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise ValueError(f"missing or empty ranges in {ranges_path}")

    ranges: list[list[int]] = []
    for item in raw_ranges:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"invalid range entry: {item!r}")
        start, end = int(item[0]), int(item[1])
        if end <= start:
            continue
        ranges.append([start, end])
    if not ranges:
        raise ValueError("no valid shot ranges")

    mp4 = resolve_mp4(video_id, metadata, fallback_lookup)
    fps = float(metadata.get("fps") or 0.0)
    if fps <= 0:
        cap = cv2.VideoCapture(mp4)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video for FPS: {mp4}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.release()

    ts_ms = np.load(ts_path)
    midpoint_frames, midpoint_timestamps, middle_jpg_by_shot, exact_ts_alignment = (
        existing_middle_mapping(source_dir, ranges, fps, ts_ms)
    )

    plans: list[dict[str, Any]] = []
    for shot_idx, ((start, end), middle_frame) in enumerate(
        zip(ranges, midpoint_frames)
    ):
        duration_sec = max(0.0, (end - start) / fps)
        total_budget, min_gap_sec, policy = shot_policy(duration_sec)
        candidates = (
            candidate_indices(
                start,
                end,
                fps=fps,
                candidate_fps=args.candidate_fps,
                middle_frame=middle_frame,
            )
            if total_budget > 1
            else []
        )
        plans.append(
            {
                "shot_index": shot_idx,
                "start_frame": start,
                "end_frame": end,
                "duration_sec": duration_sec,
                "fps": fps,
                "middle_frame": middle_frame,
                "middle_timestamp_ms": midpoint_timestamps[shot_idx],
                "total_budget": total_budget,
                "min_gap_sec": min_gap_sec,
                "policy": policy,
                "candidate_frames": candidates,
            }
        )

    # Reuse the already-written midpoint JPGs for embedding wherever possible.
    middle_path_by_frame: dict[int, Path] = {}
    for plan in plans:
        if plan["total_budget"] <= 1:
            continue
        shot_idx = int(plan["shot_index"])
        source_jpg = middle_jpg_by_shot.get(shot_idx)
        if source_jpg is not None:
            middle_path_by_frame.setdefault(int(plan["middle_frame"]), source_jpg)

    embeddings, failed_middle = embed_image_paths(
        middle_path_by_frame,
        width=int(model_args.process_width),
        height=int(model_args.process_height),
        backbone_body=backbone_body,
        device=args.device,
        batch_size=args.batch_size,
        use_amp=not args.no_amp,
    )

    # Decode only 1-FPS candidates from shots >=5s, plus any missing midpoints.
    video_embedding_indices: set[int] = set()
    for plan in plans:
        if plan["total_budget"] <= 1:
            continue
        video_embedding_indices.update(int(x) for x in plan["candidate_frames"])
        middle = int(plan["middle_frame"])
        if middle not in embeddings:
            video_embedding_indices.add(middle)

    video_embeddings, failed_video = embed_video_indices(
        mp4,
        video_embedding_indices,
        width=int(model_args.process_width),
        height=int(model_args.process_height),
        fps=fps,
        backbone_body=backbone_body,
        device=args.device,
        batch_size=args.batch_size,
        use_amp=not args.no_amp,
        sequential_gap_sec=args.sequential_gap_sec,
    )
    embeddings.update(video_embeddings)

    selected_items: list[dict[str, Any]] = []
    selection_shots: list[dict[str, Any]] = []

    for plan in plans:
        chosen = select_shot_frames(
            plan,
            embeddings=embeddings,
            threshold=args.cosine_threshold,
        )

        shot_output = {
            "shot_index": int(plan["shot_index"]),
            "range": [int(plan["start_frame"]), int(plan["end_frame"])],
            "duration_sec": round(float(plan["duration_sec"]), 6),
            "policy": plan["policy"],
            "total_budget": int(plan["total_budget"]),
            "min_gap_sec": float(plan["min_gap_sec"]),
            "candidate_count": len(plan["candidate_frames"]),
            "selected": [],
        }

        for chosen_item in chosen:
            frame_idx = int(chosen_item["frame_idx"])
            if chosen_item["kind"] == "middle":
                timestamp_ms = int(plan["middle_timestamp_ms"])
            else:
                timestamp_ms = int(round(frame_idx * 1000.0 / fps))

            record = {
                "shot_index": int(plan["shot_index"]),
                "frame_idx": frame_idx,
                "timestamp_ms": timestamp_ms,
                "kind": chosen_item["kind"],
                "max_similarity": chosen_item["max_similarity"],
                "written_to_output": (
                    args.output_mode == "all"
                    or chosen_item["kind"] == "extra"
                ),
            }
            selected_items.append(record)
            shot_output["selected"].append(record.copy())

        selection_shots.append(shot_output)

    # Keep the full selection for metadata and similarity accounting.  By
    # default, write only newly selected extras so an external embedding stage
    # does not process the already-embedded midpoint JPGs again.
    selected_items.sort(
        key=lambda item: (
            int(item["frame_idx"]),
            int(item["shot_index"]),
            0 if item["kind"] == "middle" else 1,
        )
    )
    for item in selected_items:
        item["written_to_output"] = (
            args.output_mode == "all" or item["kind"] == "extra"
        )

    output_items = [
        item for item in selected_items if bool(item["written_to_output"])
    ]

    temp_dir = destination_dir.parent / f".{video_id}.expand.tmp"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        timestamps = write_selected_frames(
            mp4,
            temp_dir,
            output_items,
            middle_jpg_by_shot=middle_jpg_by_shot,
            fps=fps,
            jpeg_quality=args.jpeg_quality,
            sequential_gap_sec=args.sequential_gap_sec,
            filename_digits=args.filename_digits,
        )

        shutil.copy2(ranges_path, temp_dir / "omnishotcut_ranges.json")

        selection_metadata = {
            "video_id": video_id,
            "mp4": mp4,
            "fps": fps,
            "source_dir": str(source_dir),
            "source_ts_count": int(len(ts_ms)),
            "range_count": int(len(ranges)),
            "source_ts_aligned_with_ranges": bool(exact_ts_alignment),
            "parameters": {
                "candidate_fps": args.candidate_fps,
                "cosine_threshold": args.cosine_threshold,
                "output_mode": args.output_mode,
                "filename_pattern": f"k_{{index:0{args.filename_digits}d}}.jpg",
                "filename_digits": args.filename_digits,
                "jpeg_quality": args.jpeg_quality,
                "batch_size": args.batch_size,
                "amp": not args.no_amp,
                "policy": {
                    "<5s": "middle only",
                    "5-<10s": "middle + at most 1 extra; minimum gap 1s",
                    "10-<30s": "middle + at most 2 extras; minimum gap 2s",
                    ">=30s": (
                        "middle + ceil(duration/15s) extras, maximum 10 total; "
                        "minimum gap 3s"
                    ),
                },
            },
            "failed_embedding_frame_indices": sorted(
                set(int(x) for x in failed_middle + failed_video)
            ),
            "output_mode": args.output_mode,
            "selection_frame_count": len(selected_items),
            "selected_middle_frame_count": sum(
                item["kind"] == "middle" for item in selected_items
            ),
            "selected_extra_frame_count": sum(
                item["kind"] == "extra" for item in selected_items
            ),
            "output_frame_count": len(output_items),
            "output_middle_frame_count": sum(
                item["kind"] == "middle" for item in output_items
            ),
            "output_extra_frame_count": sum(
                item["kind"] == "extra" for item in output_items
            ),
            "shots": selection_shots,
        }
        with open(temp_dir / DONE_FILENAME, "w", encoding="utf-8") as handle:
            json.dump(selection_metadata, handle, ensure_ascii=False, indent=2)

        if len(timestamps) != len(output_items):
            raise RuntimeError("JPG/timestamp count mismatch")

        commit_temp_dir(
            temp_dir,
            destination=destination_dir,
            source_dir=source_dir,
            overwrite=args.overwrite,
        )
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    middle_count = sum(item["kind"] == "middle" for item in output_items)
    extra_count = sum(item["kind"] == "extra" for item in output_items)
    return len(output_items), middle_count, extra_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ranges-root",
        required=True,
        help="Existing OmniShotCut keyframe root containing per-video JSON/NPY/JPG files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Output root. With extras_only it must differ from --ranges-root; "
            "with all, in-place replacement is supported."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        help="Optional fallback dataset root when the MP4 path stored in JSON is stale.",
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not access the internet; require the checkpoint to exist locally.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--candidate-fps", type=float, default=1.0)
    parser.add_argument("--cosine-threshold", type=float, default=0.94)
    parser.add_argument(
        "--output-mode",
        choices=["extras_only", "all"],
        default="extras_only",
        help=(
            "extras_only writes only newly selected frames; all writes the "
            "combined midpoint+extra set. Default: extras_only."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--filename-digits",
        type=int,
        default=6,
        help=(
            "Number of digits in output JPG names. Default 6 creates "
            "k_000001.jpg so names do not collide with existing 5-digit middle frames."
        ),
    )
    parser.add_argument(
        "--sequential-gap-sec",
        type=float,
        default=2.0,
        help="Decode sequentially instead of seeking when requested frames are this close.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--video-id-regex",
        default=r"v3c\d+_\d{5}",
        help="Regex applied to direct child folder names under --ranges-root.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.shard_count <= 0:
        raise SystemExit("--shard-count must be > 0")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must satisfy 0 <= index < count")
    if args.candidate_fps <= 0:
        raise SystemExit("--candidate-fps must be > 0")
    if not -1.0 <= args.cosine_threshold <= 1.0:
        raise SystemExit("--cosine-threshold must be within [-1, 1]")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.filename_digits < 1:
        raise SystemExit("--filename-digits must be >= 1")

    ranges_root = Path(args.ranges_root)
    out_root = Path(args.out)
    if args.output_mode == "extras_only" and same_path(ranges_root, out_root):
        raise SystemExit(
            "--output-mode=extras_only requires --out to be different from "
            "--ranges-root, otherwise the original midpoint folders would be replaced"
        )
    if not ranges_root.is_dir():
        raise SystemExit(f"ranges root not found: {ranges_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    fail_log = out_root / f"failed_expand_shard{args.shard_index}.txt"
    records = list_range_dirs(ranges_root, args.video_id_regex)
    if not records:
        raise SystemExit("no video folders containing omnishotcut_ranges.json and ts_ms.npy")

    mine = [
        path
        for index, path in enumerate(records)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit:
        mine = mine[: args.limit]

    fallback_lookup = list_videos(args.dataset_root) if args.dataset_root else {}

    print(
        f"[device] {args.device} | torch={torch.__version__} | "
        f"cuda={torch.cuda.is_available()}",
        flush=True,
    )
    print(
        f"[shard {args.shard_index}/{args.shard_count}] "
        f"{len(mine)}/{len(records)} videos | no shot inference | "
        f"output_mode={args.output_mode}",
        flush=True,
    )

    backbone_body, model_args, checkpoint_path = load_backbone_device(
        args.checkpoint_path,
        args.device,
        local_files_only=args.local_files_only,
    )
    print(
        f"[backbone] {getattr(model_args, 'backbone', 'unknown')} | "
        f"checkpoint={checkpoint_path} | "
        f"input={int(model_args.process_width)}x{int(model_args.process_height)}",
        flush=True,
    )

    started = time.time()
    done = failed = skipped = 0
    saved = middles = extras = 0

    for source_dir in mine:
        video_id = source_dir.name
        destination_dir = out_root / video_id
        marker = destination_dir / DONE_FILENAME

        if marker.exists() and not args.overwrite:
            done += 1
            skipped += 1
            continue

        try:
            frame_count, middle_count, extra_count = process_video(
                source_dir,
                destination_dir,
                fallback_lookup,
                backbone_body,
                model_args,
                args,
            )
            saved += frame_count
            middles += middle_count
            extras += extra_count
        except Exception as exc:
            failed += 1
            with open(fail_log, "a", encoding="utf-8") as handle:
                handle.write(
                    f"{video_id}\t{type(exc).__name__}: {exc}\n"
                )

        done += 1
        if done % args.log_every == 0 or done == len(mine):
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[shard {args.shard_index}] {done}/{len(mine)} videos | "
                f"frames={saved} middle={middles} extra={extras} | "
                f"skip={skipped} fail={failed} | "
                f"{done / elapsed * 60.0:.2f} vids/min",
                flush=True,
            )

    print(
        f"[done] videos={done} frames={saved} middle={middles} extra={extras} "
        f"skipped={skipped} failed={failed} out={out_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()

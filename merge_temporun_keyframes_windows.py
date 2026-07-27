from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np


DEFAULT_MIDDLE_ROOT = Path(
    r"TempoRun2026_OmniShotCut_Keyframes"
)
DEFAULT_EXTRA_ROOT = Path(
    r"TempoRun2026_OmniShotCut_Extra_Keyframes_More"
)
DEFAULT_COMBINED_ROOT = Path(
    r"TempoRun2026_OmniShotCut_Keyframes_Combined"
)


def same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def read_folder(folder: Path) -> dict[str, dict[str, object]]:
    """
    Read filename -> source path/timestamp mapping.

    Timestamp order follows sorted(k_*.jpg), matching the embedding pipeline.
    """
    if not folder.is_dir():
        return {}

    ts_path = folder / "ts_ms.npy"
    if not ts_path.is_file():
        return {}

    jpgs = sorted(folder.glob("k_*.jpg"))
    timestamps = np.load(ts_path)

    if len(jpgs) != len(timestamps):
        raise RuntimeError(
            f"JPG/ts_ms mismatch at {folder}: "
            f"jpg={len(jpgs)}, ts={len(timestamps)}"
        )

    return {
        jpg.name: {
            "source": jpg,
            "timestamp_ms": int(timestamp),
        }
        for jpg, timestamp in zip(jpgs, timestamps)
    }


def place_file(source: Path, destination: Path, mode: str) -> str:
    """
    Place an image in the combined directory.

    hardlink:
        Uses almost no extra disk space and normally works without Administrator
        rights when source and output are on the same NTFS drive.
        Falls back to copy if hardlink creation fails.

    copy:
        Creates an independent copy.

    symlink:
        Creates a symbolic link. Windows may require Developer Mode or
        Administrator rights.
    """
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"

    if mode == "symlink":
        os.symlink(source, destination)
        return "symlink"

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def copy_optional(source: Path, destination: Path) -> None:
    if source.is_file():
        shutil.copy2(source, destination)


def merge_keyframes(
    middle_root: Path,
    extra_root: Path,
    combined_root: Path,
    mode: str,
    overwrite: bool,
) -> None:
    middle_root = middle_root.resolve()
    extra_root = extra_root.resolve()
    combined_root = combined_root.resolve()

    if not middle_root.is_dir():
        raise FileNotFoundError(f"Middle root not found: {middle_root}")
    if not extra_root.is_dir():
        raise FileNotFoundError(f"Extra root not found: {extra_root}")

    if same_path(combined_root, middle_root) or same_path(combined_root, extra_root):
        raise ValueError("Combined output must differ from both input roots.")

    if combined_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {combined_root}\n"
                "Run again with --overwrite to recreate it."
            )
        print(f"Removing old output: {combined_root}", flush=True)
        shutil.rmtree(combined_root)

    combined_root.mkdir(parents=True, exist_ok=True)

    middle_ids = {
        path.name
        for path in middle_root.iterdir()
        if path.is_dir()
    }
    extra_ids = {
        path.name
        for path in extra_root.iterdir()
        if path.is_dir()
    }
    video_ids = sorted(middle_ids | extra_ids)

    total_middle = 0
    total_extra = 0
    total_combined = 0
    duplicate_timestamps = 0
    method_counts = {
        "hardlink": 0,
        "copy": 0,
        "copy_fallback": 0,
        "symlink": 0,
    }

    try:
        for index, video_id in enumerate(video_ids, start=1):
            middle_dir = middle_root / video_id
            extra_dir = extra_root / video_id
            output_dir = combined_root / video_id
            output_dir.mkdir(parents=True, exist_ok=True)

            middle = read_folder(middle_dir)
            extra = read_folder(extra_dir)

            total_middle += len(middle)
            total_extra += len(extra)

            duplicate_names = set(middle) & set(extra)
            if duplicate_names:
                raise RuntimeError(
                    f"{video_id}: duplicate filenames between middle and extra: "
                    f"{sorted(duplicate_names)[:10]}"
                )

            combined = {}
            combined.update(middle)
            combined.update(extra)

            # Detect different filenames that point to the same video timestamp.
            timestamp_counts: dict[int, int] = {}
            for record in combined.values():
                timestamp = int(record["timestamp_ms"])
                timestamp_counts[timestamp] = timestamp_counts.get(timestamp, 0) + 1

            duplicate_timestamps += sum(
                count - 1 for count in timestamp_counts.values() if count > 1
            )

            # Must match sorted(k_*.jpg), exactly like load_frame_docs().
            ordered_names = sorted(combined)
            ordered_timestamps: list[int] = []

            for filename in ordered_names:
                record = combined[filename]
                source = Path(record["source"])
                destination = output_dir / filename

                used_method = place_file(source, destination, mode)
                method_counts[used_method] += 1
                ordered_timestamps.append(int(record["timestamp_ms"]))

            np.save(
                output_dir / "ts_ms.npy",
                np.asarray(ordered_timestamps, dtype=np.int32),
            )

            # Preserve useful metadata without overwriting files from the other root.
            copy_optional(
                middle_dir / "omnishotcut_ranges.json",
                output_dir / "omnishotcut_ranges.json",
            )
            copy_optional(
                middle_dir / "keyframe_selection.json",
                output_dir / "middle_keyframe_selection.json",
            )
            copy_optional(
                extra_dir / "keyframe_selection.json",
                output_dir / "extra_keyframe_selection.json",
            )

            total_combined += len(combined)

            if index % 250 == 0 or index == len(video_ids):
                print(
                    f"[{index}/{len(video_ids)}] "
                    f"middle={total_middle:,} | "
                    f"extra={total_extra:,} | "
                    f"combined={total_combined:,}",
                    flush=True,
                )

    except Exception:
        print(
            f"\nMerge failed. Partial output remains at:\n{combined_root}",
            flush=True,
        )
        raise

    print("\nCompleted")
    print(f"Combined root: {combined_root}")
    print(f"Videos: {len(video_ids):,}")
    print(f"Middle frames: {total_middle:,}")
    print(f"Extra frames: {total_extra:,}")
    print(f"Combined frames: {total_combined:,}")
    print(f"Duplicate video/timestamp rows: {duplicate_timestamps:,}")
    print(
        "Placement methods: "
        + ", ".join(f"{key}={value:,}" for key, value in method_counts.items())
    )


def verify_combined(root: Path) -> None:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    video_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    total_jpg = 0
    total_ts = 0
    bad: list[tuple[str, int, int]] = []

    for video_dir in video_dirs:
        jpgs = sorted(video_dir.glob("k_*.jpg"))
        ts_path = video_dir / "ts_ms.npy"

        if not ts_path.is_file():
            bad.append((video_dir.name, len(jpgs), -1))
            continue

        timestamps = np.load(ts_path)
        total_jpg += len(jpgs)
        total_ts += len(timestamps)

        if len(jpgs) != len(timestamps):
            bad.append((video_dir.name, len(jpgs), len(timestamps)))

    print(f"Verified videos: {len(video_dirs):,}")
    print(f"JPG files: {total_jpg:,}")
    print(f"Timestamps: {total_ts:,}")
    print(f"Bad folders: {len(bad):,}")

    if bad:
        print("Examples:", bad[:20])
        raise RuntimeError("Combined dataset verification failed.")

    print("Verification passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge midpoint and extra OmniShotCut keyframe folders."
    )
    parser.add_argument(
        "--middle-root",
        type=Path,
        default=DEFAULT_MIDDLE_ROOT,
    )
    parser.add_argument(
        "--extra-root",
        type=Path,
        default=DEFAULT_EXTRA_ROOT,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_COMBINED_ROOT,
    )
    parser.add_argument(
        "--mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help=(
            "hardlink uses almost no extra space and falls back to copy; "
            "copy is independent; symlink may require Windows Developer Mode."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate an existing output directory.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an already-created combined directory.",
    )
    args = parser.parse_args()

    if args.verify_only:
        verify_combined(args.out)
        return

    merge_keyframes(
        middle_root=args.middle_root,
        extra_root=args.extra_root,
        combined_root=args.out,
        mode=args.mode,
        overwrite=args.overwrite,
    )
    verify_combined(args.out)


if __name__ == "__main__":
    main()

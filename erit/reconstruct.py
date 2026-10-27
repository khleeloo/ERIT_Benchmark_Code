"""Reconstruct the canonical ERIT benchmark from an authorized ElderReact copy.

The canonical metadata records the selected source-frame index in each image
filename. This script extracts exactly that frame with OpenCV, matching the
extraction method used by the original ERIT construction pipeline.

No ElderReact source media are distributed with this repository.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import cv2
import pandas as pd

SPLITS = ("train", "dev", "test")
FRAME_RE = re.compile(r"^(?P<video>.+)_ensemble_(?P<frame>\d+)\.jpg$")


def parse_frame_name(filename: str) -> tuple[str, int]:
    match = FRAME_RE.match(filename)
    if not match:
        raise ValueError(
            f"Invalid canonical ERIT frame filename: {filename!r}. "
            "Expected <video_name>_ensemble_<frame_idx>.jpg."
        )
    return match.group("video"), int(match.group("frame"))


def resolve_split_dir(elderreact_root: Path, split: str) -> Path:
    candidates = (
        elderreact_root / "ElderReact_Data" / f"ElderReact_{split}",
        elderreact_root / f"ElderReact_{split}",
        elderreact_root / split,
    )
    existing = [p for p in candidates if p.is_dir()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise RuntimeError(
            f"Multiple possible ElderReact {split!r} directories found: "
            + ", ".join(str(p) for p in existing)
        )
    raise FileNotFoundError(
        f"Could not locate ElderReact {split!r} split beneath {elderreact_root}. "
        "Expected ElderReact_Data/ElderReact_<split>, ElderReact_<split>, or <split>."
    )


def build_video_index(split_dir: Path) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for path in sorted(split_dir.rglob("*.mp4")):
        if path.stem in videos:
            duplicates.setdefault(path.stem, [videos[path.stem]]).append(path)
        else:
            videos[path.stem] = path

    if duplicates:
        detail = "; ".join(
            f"{name}: {[str(p) for p in paths]}"
            for name, paths in sorted(duplicates.items())
        )
        raise RuntimeError(f"Duplicate video stems found in {split_dir}: {detail}")

    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in {split_dir}")

    return videos


def extract_frame(video_path: Path, frame_idx: int, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or (n_frames > 0 and frame_idx >= n_frames):
        cap.release()
        raise IndexError(
            f"Frame {frame_idx} is outside {video_path.name} "
            f"(reported frame count: {n_frames})."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not decode frame {frame_idx} from {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not write image: {output_path}")


def reconstruct_split(
    split: str,
    elderreact_root: Path,
    metadata_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> tuple[int, int]:
    manifest_path = metadata_dir / f"{split}.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing canonical manifest: {manifest_path}")

    df = pd.read_csv(manifest_path, keep_default_na=False)
    if "url" not in df.columns:
        raise ValueError(f"{manifest_path} has no url column")

    split_video_dir = resolve_split_dir(elderreact_root, split)
    video_index = build_video_index(split_video_dir)

    split_out = output_dir / split
    split_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, split_out / f"{split}.csv")

    unique_frames = df["url"].drop_duplicates().tolist()
    extracted = 0

    for i, filename in enumerate(unique_frames, start=1):
        video_name, frame_idx = parse_frame_name(filename)
        video_path = video_index.get(video_name)
        if video_path is None:
            raise FileNotFoundError(
                f"Source video {video_name!r} required by {split}/{filename} "
                f"was not found in {split_video_dir}"
            )

        destination = split_out / filename
        if destination.exists() and not overwrite:
            continue

        extract_frame(video_path, frame_idx, destination)
        extracted += 1

        if i == 1 or i % 100 == 0 or i == len(unique_frames):
            print(f"[{split}] {i}/{len(unique_frames)} canonical frames processed")

    return len(df), extracted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct canonical ERIT frames from an authorized ElderReact copy."
    )
    parser.add_argument(
        "--elderreact-root",
        type=Path,
        required=True,
        help="Root of an authorized local ElderReact copy.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "metadata",
        help="Directory containing train.csv, dev.csv and test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for reconstructed ERIT.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Splits to reconstruct (default: train dev test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract images that already exist.",
    )
    args = parser.parse_args()

    args.elderreact_root = args.elderreact_root.expanduser().resolve()
    args.metadata_dir = args.metadata_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    total_rows = 0
    total_extracted = 0

    for split in args.splits:
        rows, extracted = reconstruct_split(
            split=split,
            elderreact_root=args.elderreact_root,
            metadata_dir=args.metadata_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        total_rows += rows
        total_extracted += extracted
        print(f"[{split}] rows={rows}, newly_extracted={extracted}")

    print(
        f"Reconstruction complete: {total_rows} manifest rows; "
        f"{total_extracted} images newly extracted."
    )


if __name__ == "__main__":
    main()

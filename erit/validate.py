"""Validate the canonical ERIT manifests and an optional reconstruction."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SPLITS = ("train", "dev", "test")
EXPECTED_ROWS = {"train": 1169, "dev": 634, "test": 576}
EXPECTED_VIDEOS = {"train": 602, "dev": 348, "test": 348}
EXPECTED_TOTAL_ROWS = 2379
EXPECTED_TOTAL_VIDEOS = 1298
FRAME_RE = re.compile(r"^(?P<video>.+)_ensemble_(?P<frame>\d+)\.jpg$")
KNOWN_TEST_DUPLICATE = "theywereborn_4_ensemble_105.jpg"
EXPECTED_DEV_EMPTY_ORIGINAL_LABELS = 7


def source_video(filename: str) -> str:
    match = FRAME_RE.match(filename)
    if not match:
        raise ValueError(filename)
    return match.group("video")


def load_manifests(metadata_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = metadata_dir / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames[split] = pd.read_csv(path, keep_default_na=False)
    return frames


def validate_metadata(metadata_dir: Path) -> dict[str, pd.DataFrame]:
    frames = load_manifests(metadata_dir)
    required = {
        "url",
        "instruction",
        "OriginalLabels",
        "caption",
        "GTEmotionScore",
        "whisper_mean_conf",
        "whisper_n_low_conf",
    }

    split_videos: dict[str, set[str]] = {}

    for split, df in frames.items():
        missing_cols = required - set(df.columns)
        assert not missing_cols, f"{split}: missing columns {sorted(missing_cols)}"
        assert len(df) == EXPECTED_ROWS[split], (
            f"{split}: expected {EXPECTED_ROWS[split]} rows, found {len(df)}"
        )

        bad_names = [u for u in df["url"] if FRAME_RE.match(str(u)) is None]
        assert not bad_names, f"{split}: invalid canonical filenames: {bad_names[:5]}"

        split_videos[split] = {source_video(str(u)) for u in df["url"]}
        assert len(split_videos[split]) == EXPECTED_VIDEOS[split], (
            f"{split}: expected {EXPECTED_VIDEOS[split]} source videos, "
            f"found {len(split_videos[split])}"
        )

    total_rows = sum(len(df) for df in frames.values())
    assert total_rows == EXPECTED_TOTAL_ROWS, (
        f"Expected {EXPECTED_TOTAL_ROWS} total rows, found {total_rows}"
    )

    total_videos = sum(len(v) for v in split_videos.values())
    assert total_videos == EXPECTED_TOTAL_VIDEOS, (
        f"Expected {EXPECTED_TOTAL_VIDEOS} split-local source videos, found {total_videos}"
    )

    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = split_videos[a] & split_videos[b]
        assert not overlap, f"Source-video leakage between {a} and {b}: {sorted(overlap)}"

    dev_empty = frames["dev"]["OriginalLabels"].astype(str).str.strip().eq("")
    assert int(dev_empty.sum()) == EXPECTED_DEV_EMPTY_ORIGINAL_LABELS, (
        f"dev: expected exactly {EXPECTED_DEV_EMPTY_ORIGINAL_LABELS} blank "
        f"OriginalLabels rows, found {int(dev_empty.sum())}"
    )
    assert frames["dev"].loc[dev_empty, "caption"].eq("Neutral").all(), (
        "dev: all blank OriginalLabels rows are expected to be Neutral"
    )

    test_dups = frames["test"][frames["test"].duplicated(keep=False)]
    assert len(test_dups) == 2, (
        "test: canonical release is expected to contain exactly one duplicated row"
    )
    assert test_dups["url"].eq(KNOWN_TEST_DUPLICATE).all(), (
        "test: unexpected duplicated row"
    )

    print("Canonical ERIT metadata validation passed.")
    print(
        "Rows: "
        + ", ".join(f"{s}={len(frames[s])}" for s in SPLITS)
        + f", total={total_rows}"
    )
    print(
        "Source videos: "
        + ", ".join(f"{s}={len(split_videos[s])}" for s in SPLITS)
        + f", total={total_videos}"
    )
    print(
        f"Known release characteristics: {EXPECTED_DEV_EMPTY_ORIGINAL_LABELS} "
        "blank dev OriginalLabels rows; 1 duplicated test row."
    )

    return frames


def validate_reconstruction(
    benchmark_root: Path, frames: dict[str, pd.DataFrame]
) -> None:
    for split, df in frames.items():
        split_dir = benchmark_root / split
        csv_path = split_dir / f"{split}.csv"
        assert csv_path.is_file(), f"Missing reconstructed manifest: {csv_path}"

        expected_images = set(df["url"])
        actual_images = {p.name for p in split_dir.glob("*.jpg")}

        missing = expected_images - actual_images
        extra = actual_images - expected_images
        assert not missing, (
            f"{split}: {len(missing)} missing images; sample={sorted(missing)[:5]}"
        )
        assert not extra, (
            f"{split}: {len(extra)} unexpected images; sample={sorted(extra)[:5]}"
        )

        print(
            f"{split}: reconstruction contains all {len(expected_images)} "
            "unique canonical images"
        )

    print("Reconstructed ERIT validation passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ERIT metadata/reconstruction.")
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "metadata",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=None,
        help="Optional reconstructed ERIT root to validate.",
    )
    args = parser.parse_args()

    metadata_dir = args.metadata_dir.expanduser().resolve()
    frames = validate_metadata(metadata_dir)

    if args.benchmark_root is not None:
        validate_reconstruction(args.benchmark_root.expanduser().resolve(), frames)


if __name__ == "__main__":
    main()

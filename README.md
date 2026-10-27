# ERIT Benchmark

Code and metadata for reconstructing **ERIT: A Responsible Multimodal Affective Computing Benchmark for Older-Adult Emotion Recognition in the Wild** from an authorized local copy of ElderReact.

ERIT contains fixed image-text pairs for controlled multimodal emotion-recognition evaluation. This repository does **not** redistribute ElderReact videos or extracted source media.

## Dataset

| Split | Samples | Source videos |
|---|---:|---:|
| Train | 1,169 | 602 |
| Dev | 634 | 348 |
| Test | 576 | 348 |
| **Total** | **2,379** | **1,298** |

Emotion classes: **Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral**.

## Repository layout

```text
.
├── metadata/
│   ├── train.csv
│   ├── dev.csv
│   ├── test.csv
│   └── README.md
├── erit/
│   ├── __init__.py
│   ├── reconstruct.py
│   └── validate.py
├── LICENSE
├── requirements.txt
└── README.md
```

## Source data

ERIT is derived from ElderReact. Source media are **not included** in this repository.

To reconstruct ERIT, obtain ElderReact through its authorized distribution channel and retain its original split directories. The reconstruction script supports:

```text
ElderReact/
└── ElderReact_Data/
    ├── ElderReact_train/
    ├── ElderReact_dev/
    └── ElderReact_test/
```

Users remain responsible for complying with the original ElderReact access and usage conditions.

## Installation

```bash
git clone https://github.com/khleeloo/ERIT_Benchmark_Code.git
cd ERIT_Benchmark_Code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Validate the canonical metadata

```bash
python -m erit.validate --metadata-dir metadata
```

The validator checks the published split sizes, 2,379 total rows, 1,298 source videos, zero source-video overlap, valid canonical frame names, and the known release characteristics documented in `metadata/README.md`.

## Reconstruct ERIT

```bash
python -m erit.reconstruct \
    --elderreact-root /path/to/ElderReact \
    --metadata-dir metadata \
    --output-dir /path/to/ERIT
```

A canonical filename such as `play_road_redemp_59_ensemble_645.jpg` identifies frame 645 of `play_road_redemp_59.mp4`. Frames are extracted with OpenCV using the same extraction method as the original ERIT frame-selection pipeline.

After reconstruction:

```bash
python -m erit.validate \
    --metadata-dir metadata \
    --benchmark-root /path/to/ERIT
```

## Reproducibility note

The CSV manifests are the **canonical benchmark definition**. Re-running the original DeepFace/CLIP/ViT-FER selection models can change scores or selected frames as model packages and inference environments evolve. Benchmark reconstruction therefore uses the fixed frame indices recorded in the manifests.

## Responsible use

ERIT is a research benchmark for analyzing multimodal model limitations. It is not intended for deployment, surveillance, clinical assessment, or individual profiling.

## Citation

Rita Frieske and Bertram E. Shi. *ERIT: A Responsible Multimodal Affective Computing Benchmark for Older-Adult Emotion Recognition in the Wild.* MRAC 2026, ACM Multimedia 2026. DOI: 10.1145/3840474.3840554

Please also cite the original ElderReact dataset.

## License

The repository license applies to ERIT-authored code and metadata only. ElderReact source media are not redistributed and remain subject to their original terms.

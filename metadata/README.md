# Canonical ERIT manifests

These CSV files define the ERIT benchmark used in the published experiments.

## Columns

- `url`: canonical selected-frame filename. The suffix `_ensemble_<N>.jpg` records the zero-based frame index `N` in the corresponding ElderReact source video.
- `instruction`: Whisper Large V3 transcript associated with the source video.
- `OriginalLabels`: original ElderReact emotion labels when present in the source metadata.
- `caption`: target emotion for this ERIT sample.
- `GTEmotionScore`: ensemble consensus score for the selected target-emotion frame.
- `whisper_mean_conf`: mean Whisper word-confidence estimate.
- `whisper_n_low_conf`: number of transcript words below the 0.7 confidence threshold.

## Canonical release characteristics

- train: 1,169 rows from 602 videos;
- dev: 634 rows from 348 videos;
- test: 576 rows from 348 videos;
- total: 2,379 rows from 1,298 videos;
- zero source-video overlap across train/dev/test;
- seven dev Neutral rows have blank `OriginalLabels`;
- one test row (`theywereborn_4_ensemble_105.jpg`, Happiness) is duplicated exactly;
- multiple rows may legitimately reference the same image when one selected frame is used for more than one valid target emotion.

Do not deduplicate or rewrite these manifests if exact comparability with the published ERIT benchmark is required.

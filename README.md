# ERIT: ElderReact Image-Text Benchmark

ERIT is a multimodal emotion recognition benchmark built from ElderReact videos.
Each sample pairs:

- one representative frame selected with an ensemble strategy, and
- one Whisper transcript (with confidence metadata).

This repository contains the current code used for dataset construction,
zero-shot/fine-tuned evaluation, robustness analysis, and statistical testing.

## What ERIT Evaluates

ERIT is designed as a benchmark suite along three axes:

- modality: image-only (I), text-only (T), and fusion (I+T)
- model regime: zero-shot and LoRA fine-tuned
- robustness: controlled perturbations (grayscale, blurred face, keyword removal,
	text dropout, image dropout)

Primary reported metric is multi-label accuracy (MAcc), with additional:

- accuracy (Acc)
- balanced accuracy (BAcc)
- macro F1

## Dataset Summary (from the manuscript)

- total samples: 2,379 image-text pairs
- unique source videos: 1,298
- emotion classes: 7
- multi-label samples: 1,783 (74.9%)
- single-label samples: 596 (25.1%)
- mean Whisper confidence: 0.87 +/- 0.13

## Repository Structure

- dataset build and preprocessing:
	- `erit/build_erit_new.py`
	- `erit/whisper_transcribe_elderreact.py`
	- `erit/generate_perturbations.py`
- model evaluation:
	- `experiments/eval_benchmark.py`
	- `experiments/eval_benchmark_qwen2vl.py`
	- `experiments/eval_llava.py`
	- `experiments/eval_qwen2vl.py`
	- `experiments/eval_llava_lora.py`
	- `experiments/finetune_llava_lora.py`
- analysis and reporting:
	- `experiments/analyze_benchmark.py`
	- `experiments/confusion_analysis.py`
	- `experiments/frame_selection_ablation.py`
	- `erit/ensemble_frame_selection.py`
	- `experiments/eval_metrics_extended.py`
	- `experiments/eval_ablation.py`
	- `experiments/statistical_tests.py`
- paper/source text:
	- `benchmark_methodology.tex`
	- `benchmark_results.tex`

## Installation

```bash
pip install -r requirements.txt
```

## Typical Workflow

1. Transcribe ElderReact audio:

```bash
python erit/whisper_transcribe_elderreact.py
```

2. Build ERIT metadata / benchmark tables:

```bash
python erit/build_erit_new.py
```

3. Generate robustness perturbations:

```bash
python erit/generate_perturbations.py
```

4. Run zero-shot benchmark:

```bash
bash scripts/run_eval_models.sh
```

5. Run extended/ablation analyses:

```bash
bash scripts/run_eval_extended.sh
bash scripts/run_analysis.sh
```

For cluster runs, use the provided `jobs/slurm_*.sbatch` scripts.

## Key Findings (from benchmark_results.tex)

- Fusion outperforms single-modality settings across evaluated models.
- LoRA fine-tuning on ERIT yields strong gains over zero-shot baselines.
- Fusion is resilient to mild perturbations (keyword removal, grayscale).
- Severe visual degradation (image dropout / heavy blur) causes the largest
	performance drops, indicating strong visual dependence.
- Stratified analysis shows high conflict rates between image-only and text-only
	predictions, where fusion frequently recovers performance.

## Citation

If you use ERIT, please cite the associated article and ElderReact:

- Ma et al., ElderReact (2019)
- ERIT benchmark paper (see manuscript text in this repo)

## Notes

- This README reflects the up-to-date code currently maintained in this repository.
- Older legacy scripts from previous repo versions have been removed.

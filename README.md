# Focused long-document classification study

This repository implements the shared from-scratch harness described in [PLAN.md](PLAN.md): exact attention, Longformer, Linformer, Long-Short, Compressive Transformer, and RoFormer are trained with one tokenizer, one model scale, one MLM token budget, and task-correct evaluation. The four real datasets keep separate heads and metrics; their labels are never pooled.

For the concise setup and command reference—including the initial head-to-head suite, shared stress chamber, live checklist, server resume behavior, and presentation artifacts—see [EXPERIMENTS.md](EXPERIMENTS.md). The longer operational notes remain in [RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md).

## Install

Python 3.10+ and PyTorch 2.2+ are required. Create an environment using the PyTorch wheel appropriate for your CUDA version, then install this project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# If needed, install the CUDA-specific torch wheel from https://pytorch.org/get-started/locally/
python -m pip install -e '.[dev]'
pytest
```

The repository also includes a `setup.py` compatibility shim for older pip
versions that do not invoke the PEP 660 editable-install hook. Upgrading the
three packaging tools first is still recommended so editable installs use the
modern `pyproject.toml` path.

The harness runs on CPU, CUDA, and MPS. CUDA automatically uses BF16 when supported (FP16 otherwise), TF32 matrix multiplication, fused AdamW, pinned transfers, PyTorch scaled-dot-product attention, fully packed MLM batches, masked-position-only vocabulary projection, and gradient checkpointing. On GPUs with enough memory, `--set train.gradient_checkpointing=false` avoids recomputation without changing the scientific configuration and is safe when resuming a checkpoint. Set `train.compile=true` only after validating your installed PyTorch/CUDA combination; custom sparse operators do not benefit equally from compilation.

## Prepare all data at a chosen location

Dataset preparation is deliberately separate from training:

```bash
python download_data.py --data-dir /path/to/longdoc-data
```

This downloads:

- `ccdv/arxiv-classification` (`no_ref`, authoritative train/validation/test split);
- LexGLUE SCOTUS and ECtHR-A with their official splits;
- IMDB with its official test set and a fixed, stratified 2,500-example validation subset taken only from training; duplicate training documents and reviews that overlap test are removed before splitting.

It loads immutable Hugging Face Parquet snapshots (so it is compatible with modern `datasets` releases), keeps the download cache under `/path/to/longdoc-data/cache`, normalizes each dataset under `/path/to/longdoc-data/datasets`, removes exact cross-split overlaps with priority `test > validation > train`, verifies split isolation, writes the cleanup counts and provenance manifests, and trains the shared 32K byte-level BPE at `/path/to/longdoc-data/tokenizer/tokenizer.json`. Only training documents are exposed to tokenizer learning. The MLM iterator likewise reads only training splits and balances corpora and context lengths by processed non-padding tokens.

Useful preparation options:

```bash
# Prepare a subset while debugging
python download_data.py --data-dir /path/to/data --datasets imdb scotus

# Rebuild normalized data and tokenizer
python download_data.py --data-dir /path/to/data --force
```

The complete download is several GB. Do not use `--force` unless replacing the prepared copies at that exact data path is intended.

## Run an experiment

List registered baselines and novelty variants:

```bash
python run_experiment.py --stage list
```

Run every experiment and ablation associated with one paper:

```bash
run_experiment \
  --experiment longformer \
  --data-dir /path/to/longdoc-data \
  --output-dir /path/to/results
```

The five accepted paper names are `longformer`, `linformer`, `longshort`, `compressive`, and `roformer`. The runner resumes automatically using a persistent manifest and full-state checkpoints. Use `--stage plan` to inspect all jobs without running them.

Run the two shared comparisons end to end:

```bash
run_experiment --experiment head_to_head --run-id main \
  --data-dir /path/to/longdoc-data --output-dir /path/to/results
run_experiment --experiment stress_test --run-id main \
  --reuse-pretraining-from /path/to/results/head_to_head/main \
  --data-dir /path/to/longdoc-data --output-dir /path/to/results
```

Each suite maintains `status/CHECKLIST.md` and `status/status_summary.json`; use `--stage status` with the same experiment, run ID, config, and paths to refresh and print the snapshot.

Run one registry cell for debugging with `--cell`:

```bash
run_experiment --cell longformer.learned_global --stage all --task arxiv \
  --data-dir /path/to/longdoc-data --output-dir /path/to/results
```

Low-level `all` resolves checkpoint reuse automatically. For example, `longformer.learned_global` first looks for `longformer.baseline/pretrain/checkpoint_last.pt`, because only the new selector should be learned during fine-tuning. In contrast, `longshort.segment_summaries` and `compressive.dual_channel` get fresh MLM checkpoints because their encoder operators changed. Pass `--checkpoint PATH` to use an explicit checkpoint.

Run all four real classification tasks without mixing labels or metrics:

```bash
run_experiment \
  --cell roformer.baseline --stage finetune --task all \
  --data-dir /path/to/longdoc-data --output-dir /path/to/results
```

Run the shared four-challenge stress chamber:

```bash
run_experiment --cell linformer.resampled_exact_cls --stage all \
  --task stress --data-dir /path/to/longdoc-data
```

`stress` expands to four independently fine-tuned binary tasks at exact 1K, 2K, and 4K model-token lengths:

| Challenge | Controlled change | Diagnosis |
|---|---|---|
| `stress_retrieve` | One label cue at 5/25/50/75/95% | Positional coverage |
| `stress_combine` | Two cues at least half a document apart; label is XOR | Multi-evidence integration |
| `stress_order` | Label is whether alpha precedes beta | Preservation of order |
| `stress_distract` | 0, 8, or 32 irrelevant markers around the cue | Selection/compression robustness |

Train, validation, and test use disjoint complete marker phrases. Class-role words remain interpretable across templates, avoiding an impossible arbitrary-symbol generalization problem. Test metrics include length, position, region, and distractor-count breakdowns.

### Registered study cells

| Family | Baseline / primary novelty cells |
|---|---|
| Control | `exact.baseline`, `exact.truncated512` |
| Longformer | `longformer.baseline`, `longformer.learned_global`, `longformer.random_global`, `longformer.even_global`, `longformer.sentence_global` |
| Linformer | `linformer.fixed`, `linformer.fixed_exact_cls`, `linformer.resampled`, `linformer.resampled_exact_cls` |
| Long-Short | `longshort.baseline`, `longshort.segment_summaries` |
| Compressive | `compressive.baseline`, `compressive.salient`, `compressive.dual_channel`, `compressive.dual_channel_full_bptt` |
| RoFormer | `roformer.baseline`, `roformer.frequency_gates` |

Use dotted overrides for matched ablations or smaller pilots:

```bash
run_experiment --cell linformer.resampled --stage all --task imdb \
  --data-dir /path/to/data \
  --set model.projection_rank=32 --set train.mlm_tokens=10000000
```

For RoFormer train-short/test-long evaluation, keep model capacity at 4K, train and select the checkpoint at 1K, then evaluate that unchanged checkpoint on the test set containing 1K/2K/4K examples:

```bash
run_experiment --cell roformer.frequency_gates --stage all \
  --task stress_retrieve_extrapolation --data-dir /path/to/data \
  --set train.finetune_train_length=1024 \
  --set train.finetune_validation_length=1024 \
  --set train.finetune_eval_length=4096
```

### Pretraining-compute curves

The full configuration saves architecture checkpoints at 25M, 50M, 100M, and 200M processed non-padding tokens, independent of optimizer-step count. Fine-tune every milestone with:

```bash
run_experiment --cell linformer.fixed --stage curve --task all \
  --data-dir /path/to/data --output-dir /path/to/results
```

This writes `compute_curve/tokens_<budget>/...`. `--stage summarize` combines those results with measured pretraining GPU-hours, 4K throughput, and peak memory. The endpoint answers equal-data quality at 200M tokens; the curve plus benchmark answers the efficiency-frontier question. Equal optimizer steps are not treated as evidence of equal compute.

`configs/base.yaml` is the paper-scale configuration: 4 layers, width 512, FFN 2,048, 8 heads, 4,096 tokens, and 200M non-padding MLM tokens. `configs/quick.yaml` is an end-to-end plumbing check, not a publishable training budget.

## Baselines, efficiency, and inference-only analyses

Run the character n-gram TF-IDF baseline:

```bash
python run_experiment.py --stage tfidf --task all --data-dir /path/to/data
```

Benchmark synchronized model compute (data loading is excluded):

```bash
run_experiment --cell compressive.dual_channel --stage benchmark \
  --device cuda --output-dir /path/to/results
```

The benchmark writes documents/second, batch latency, CUDA peak allocated memory, parameter count, and architecture-aware estimated attention FLOPs at every configured length.

Inference-only analyses take a fine-tuned checkpoint:

```bash
# Shared truncation curve
run_experiment --cell linformer.resampled --stage analyze --analysis truncation \
  --task imdb --checkpoint /path/to/checkpoint_best.pt --data-dir /path/to/data

# Linformer position shift and spectrum
run_experiment --cell linformer.resampled --stage analyze --analysis prepend \
  --analysis-value 256 --task imdb --checkpoint /path/to/checkpoint_best.pt --data-dir /path/to/data
run_experiment --cell linformer.resampled --stage analyze --analysis projection_spectrum \
  --task imdb --checkpoint /path/to/checkpoint_best.pt

# Longformer selected-position histogram and scorer quantiles (up to 100 batches)
run_experiment --cell longformer.learned_global --stage analyze \
  --analysis selection_diagnostics --task arxiv --checkpoint /path/to/checkpoint_best.pt \
  --data-dir /path/to/data

# Every-model chunk shuffling
run_experiment --cell longshort.segment_summaries --stage analyze --analysis chunk_shuffle \
  --analysis-value 512 --task arxiv --checkpoint /path/to/checkpoint_best.pt --data-dir /path/to/data
```

Architecture-specific checkpoint-preserving ablations use `--analysis ablation` plus an override:

```bash
# Long-Short summaries off
--set model.summary_type=disabled
# Long-Short region identities shuffled while content summaries stay fixed
--set model.summary_ablation=shuffle
# Compressive recent memory, compressed memory, or a dual-channel half off
--set model.memory_ablation=regular      # also: compressed, coverage, salience, current
# RoFormer low/middle/high rotary band off (0/1/2)
--set model.rope_ablate_band=0
# Learned Longformer selector evaluated with its lowest-scoring positions
--set model.global_selection=lowest
```

Scalar diagnostics such as summary/global attention mass and memory-token counts are included in analysis JSON. Fine-tuning writes raw logits and labels so paired bootstrap comparisons can be computed without another model pass.

Create the presentation-ready failure-fingerprint artifact after completing the six primary stress runs and benchmarks:

```bash
python run_experiment.py --stage fingerprint --output-dir /path/to/results
```

This writes `reports/failure_fingerprint.json` and a standalone SVG heatmap. Rows are the six controlled models; columns are late retrieval, XOR, order, 32 distractors, 1K→4K extrapolation, and throughput. Binary-task scores are normalized above chance, throughput is normalized to the fastest model, cells show seed uncertainty, and each architecture has a small attention-pattern icon.

The extrapolation column reads the distinct `stress_retrieve_extrapolation` run, which automatically constrains training and validation to 1K while retaining the 4K model capacity and evaluating at all three lengths. It never relabels an all-length-trained result as extrapolation.

## Outputs and reproducibility

Results use this layout:

```text
runs/<experiment>/
  pretrain/config.yaml
  pretrain/checkpoint_last.pt
  pretrain/checkpoint_tokens_25000000.pt
  pretrain/checkpoint_tokens_50000000.pt
  pretrain/checkpoint_tokens_100000000.pt
  pretrain/checkpoint_tokens_200000000.pt
  pretrain/train.jsonl
  finetune/<task>/seed_<seed>/
    config.yaml
    checkpoint_best.pt
    validation.jsonl
    test_metrics.json
    test_predictions.npz
  benchmark.json
  analysis/*.json
  compute_curve/tokens_*/*/seed_*/
  reports/failure_fingerprint.{json,svg}
```

Classification metrics are task-specific: accuracy and macro-F1 for arXiv/SCOTUS, micro- and macro-F1 for ECtHR-A, and accuracy for IMDB. Each test file includes a paired-bootstrap-compatible 95% interval. Summarize completed runs with:

```bash
python run_experiment.py --stage summarize --output-dir /path/to/results
```

The summary reports per-task mean ± standard deviation and mean rank across tasks; it never averages incompatible metric values. Checkpoints are written atomically, configs are copied into every run, all sampling is seeded, throughput synchronizes CUDA, and retained-token coverage is reported for every task.

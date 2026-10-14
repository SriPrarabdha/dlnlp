# Experiment quickstart

This is the short operational guide for preparing data, launching every study, checking progress, resuming a server job, and finding presentation artifacts.

## 1. Install and configure monitoring

Use Python 3.10+ and install the PyTorch wheel appropriate for the server CUDA version first.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest
nvidia-smi
```

The repository `.env` may contain `WANDB_API_KEY=...`; it is loaded without printing the key. W&B is online by default. Add `--no-wandb` for a local run or `--wandb-mode offline` during a network outage. Local logs and checkpoints remain authoritative.

## 2. Prepare data once

Training never downloads data implicitly.

```bash
longdoc-data --data-dir /mnt/data/longdoc
```

This prepares the official/fixed arXiv, SCOTUS, ECtHR-A, and IMDB splits plus the shared 32K tokenizer. Only training partitions enter tokenizer learning and MLM pretraining.

## 3. Inspect available work

```bash
run_experiment --stage list
run_experiment --experiment head_to_head --stage plan \
  --data-dir /mnt/data/longdoc --output-dir /mnt/results/longdoc
run_experiment --experiment stress_test --stage plan \
  --data-dir /mnt/data/longdoc --output-dir /mnt/results/longdoc
```

`plan` performs no training. Use `configs/quick.yaml` only to test plumbing; publishable runs use `configs/base.yaml` (the default).

## 4. Initial head-to-head comparison

```bash
run_experiment --experiment head_to_head --run-id main \
  --data-dir /mnt/data/longdoc --output-dir /mnt/results/longdoc --device cuda
```

This one command schedules:

- pooled-training-text MLM pretraining for exact, Longformer, Linformer, Long-Short, Compressive, and RoFormer at the matched 200M-token endpoint;
- arXiv, SCOTUS, ECtHR-A, and IMDB fine-tuning with three seeds and task-correct metrics;
- exact-attention-at-512 and TF-IDF sanity controls;
- 25M/50M/100M/200M accuracy-versus-pretraining-compute runs for all six headline models;
- synchronized throughput, memory, FLOP, and parameter measurements at 512/1K/2K/4K;
- per-model/per-task 512/1K/2K/4K truncation curves.

Important presentation plots are generated under `.../head_to_head/main/presentation/`: downstream score heatmap, mean task rank, efficiency frontier, throughput and memory by length, parameter count, and retained-token coverage. Generic validation, pretraining, compute-curve, and per-cell plots are under `plots/` and `cells/*/plots/`.

## 5. Shared stress chamber

Run independently (it will pretrain its own matched checkpoints):

```bash
run_experiment --experiment stress_test --run-id main \
  --data-dir /mnt/data/longdoc --output-dir /mnt/results/longdoc --device cuda
```

To avoid repeating the six expensive MLM runs after a compatible head-to-head run, import its immutable checkpoints. Compatibility of data, seed, model, and training settings is checked before work begins; hard links are used when the filesystem permits them.

```bash
run_experiment --experiment stress_test --run-id main \
  --reuse-pretraining-from /mnt/results/longdoc/head_to_head/main \
  --data-dir /mnt/data/longdoc --output-dir /mnt/results/longdoc --device cuda
```

The stress suite fine-tunes all six architectures on:

- `stress_retrieve`: cue at 5/25/50/75/95%;
- `stress_combine`: distant two-cue XOR;
- `stress_order`: whether A precedes B;
- `stress_distract`: 0/8/32 irrelevant markers;
- `stress_retrieve_extrapolation`: RoFormer trained/selected at 1K and tested unchanged at 1K/2K/4K.

It also runs XOR/order chunk-shuffle diagnostics. Its `presentation/` directory contains the normalized failure-fingerprint heatmap with uncertainty and attention icons, retrieval-position curve, distractor curve, one length curve per challenge, and the RoFormer extrapolation curve.

## 6. Five targeted paper suites

Each command runs that paper’s baseline, novelty cells, real/stress tasks, matched ablations, benchmarks, and compute curve:

```bash
run_experiment --experiment longformer  --run-id main --data-dir /mnt/data/longdoc --output-dir /mnt/results/longdoc
run_experiment --experiment linformer   --run-id main --data-dir /mnt/data/longdoc --output-dir /mnt/results/longdoc
run_experiment --experiment longshort   --run-id main --data-dir /mnt/data/longdoc --output-dir /mnt/results/longdoc
run_experiment --experiment compressive --run-id main --data-dir /mnt/data/longdoc --output-dir /mnt/results/longdoc
run_experiment --experiment roformer    --run-id main --data-dir /mnt/data/longdoc --output-dir /mnt/results/longdoc
```

Use `--reuse-pretraining-from PATH_TO_COMPATIBLE_SUITE_RUN` where that run contains the required baseline cell checkpoint. Novel cells that alter the encoder and require fresh pretraining still need their matching checkpoint in the source; otherwise the suite stops with the exact missing path.

## 7. Live checklist, results, and resume

Every state change atomically regenerates:

```text
/mnt/results/longdoc/<suite>/<run-id>/status/CHECKLIST.md
/mnt/results/longdoc/<suite>/<run-id>/status/status_summary.json
```

The checklist separates running, completed, pending, failed, and dependency-blocked sub-experiments. Completed fine-tuning rows show metric mean ± standard deviation and seed count; benchmark rows show throughput and memory. Inspect or refresh it from the CLI:

```bash
run_experiment --experiment head_to_head --run-id main --stage status \
  --data-dir /mnt/data/longdoc --output-dir /mnt/results/longdoc
watch -n 10 cat /mnt/results/longdoc/head_to_head/main/status/CHECKLIST.md
```

Resume an interrupted/preempted run by repeating its exact command. The manifest skips verified artifacts, while active trainers restore model, optimizer, scheduler, AMP scaler, RNG, sampler, epoch/batch, and token counters. `SIGINT`, `SIGTERM`, and Slurm `SIGUSR1` request an emergency checkpoint. Logs contain rolling ETAs.

```bash
# Slurm: request a warning two minutes before termination
#SBATCH --signal=B:USR1@120
srun run_experiment --experiment stress_test --run-id main ...

# Fail instead of silently starting a missing run/checkpoint
run_experiment --experiment stress_test --run-id main --resume required ...
```

Use the same config, paths, overrides, and run ID when resuming. Use `--new-run` only when a deliberately separate replicate is wanted.

## 8. Output map

```text
<output>/<suite>/<run-id>/
  manifest.json                 # authoritative scheduler state and errors
  resolved_config.yaml
  suite_plan.yaml
  status/{CHECKLIST.md,status_summary.json}
  logs/{run.log,events.jsonl}
  cells/<cell>/pretrain/        # resumable MLM state and milestones
  cells/<cell>/finetune/<task>/seed_<seed>/
  cells/<cell>/benchmark/
  curves/<cell>/<task>/tokens_<budget>/
  analyses/<cell>/<ablation>/seed_<seed>/
  baselines/tfidf/<task>/
  summaries/                    # compact JSON result tables
  plots/                        # generic study figures
  presentation/                 # curated slide-ready SVG figures
```

Checkpoint and status writes are atomic, logs rotate, JSONL events are append-only, W&B run IDs are stable, and `.suite.lock` prevents two processes from writing the same suite run.

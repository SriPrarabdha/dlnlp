# Setup and run the five paper suites

For the two shared study entry points (`head_to_head` and `stress_test`), the live checklist, cross-suite checkpoint reuse, and the concise command reference, use [EXPERIMENTS.md](EXPERIMENTS.md). This file focuses on the five targeted paper suites.

## 1. Install

Use Python 3.10 or newer. Install the PyTorch build that matches the server CUDA driver, then install this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

This installs the `run_experiment` and `longdoc-data` commands. W&B, dotenv loading, and server-process monitoring are included in the normal dependencies.

The repository `.env` file may contain:

```dotenv
WANDB_API_KEY=...
```

The runner loads that file without printing the key. W&B uses online mode by default. Use `--no-wandb` for a local-only debug run or `--wandb-mode offline` when a server temporarily has no network access.

Local checkpoints and logs remain authoritative if W&B is temporarily unavailable. To make a monitoring connection mandatory, add `--set monitoring.required=true`.

## 2. Prepare all data at an explicit location

Data preparation is a separate command and never runs implicitly during training:

```bash
longdoc-data --data-dir /mnt/datasets/longdoc
```

The chosen directory contains normalized arXiv, SCOTUS, ECtHR-A, and IMDB splits, download provenance, caches, and the shared 32K byte-level BPE tokenizer. Only training partitions enter tokenizer training or pooled MLM pretraining.

## 3. Inspect a suite before launching it

The only valid paper-suite names are `longformer`, `linformer`, `longshort`, `compressive`, and `roformer`:

```bash
run_experiment --stage list

run_experiment \
  --experiment longformer \
  --stage plan \
  --data-dir /mnt/datasets/longdoc \
  --output-dir /mnt/results/longdoc
```

The plan lists every pretraining, fine-tuning, compute-curve, benchmark, and inference-ablation job before any GPU work starts.

## 4. Run one paper and all of its experiments

`suite` is the default stage, so the minimal server command is:

```bash
run_experiment \
  --experiment longformer \
  --data-dir /mnt/datasets/longdoc \
  --output-dir /mnt/results/longdoc \
  --device cuda
```

Replace `longformer` with any of the other four names. A full configuration means 200M non-padding MLM tokens, all required real/stress tasks, three fine-tuning seeds, compute checkpoints, throughput measurements, and paper-specific ablations. Use `--config configs/quick.yaml` only for an end-to-end plumbing check; it is not a research-quality budget.

The terminal and log file show a rolling ETA. W&B groups all jobs under `<paper>/<run-id>` and gives each task, seed, benchmark, and ablation a stable resumable run identity.

On a high-memory GPU, disable activation checkpointing to trade memory for
throughput while keeping the same token batches and optimizer schedule:

```bash
run_experiment \
  --experiment head_to_head --run-id main \
  --data-dir /mnt/datasets/longdoc \
  --output-dir /mnt/results/longdoc \
  --device cuda \
  --set train.gradient_checkpointing=false
```

This compute-only setting may be changed when resuming. Increasing
`train.batch_tokens`, in contrast, changes the number of optimizer updates for
the fixed MLM token budget and should be treated as a new scientific run.

For Slurm, ask the scheduler to warn the process before its time limit so the signal-aware checkpoint path runs:

```bash
#SBATCH --signal=B:USR1@120
srun run_experiment --experiment longformer \
  --data-dir /mnt/datasets/longdoc --output-dir /mnt/results/longdoc
```

For a plain SSH server, run inside `tmux`/`screen`, or redirect the console while retaining the structured logs:

```bash
nohup run_experiment --experiment longformer \
  --data-dir /mnt/datasets/longdoc --output-dir /mnt/results/longdoc \
  > longformer.console.log 2>&1 &
```

## 5. Resume after interruption or preemption

Resume is automatic. The default run ID is a hash of the resolved configuration, so running the identical command again opens the same suite manifest and skips verified completed artifacts:

```bash
run_experiment \
  --experiment longformer \
  --data-dir /mnt/datasets/longdoc \
  --output-dir /mnt/results/longdoc
```

Training checkpoints contain model, optimizer, scheduler, AMP scaler, RNG, token-packer state, epoch, and batch progress. `SIGTERM`, `SIGINT`, and `SIGUSR1` request an emergency checkpoint before exit. Periodic checkpoints and heartbeat files are also written while a job is healthy.

`SIGTERM` and `SIGUSR1` also mark the active W&B run as preempting before it exits; `SIGINT` is treated as a user cancellation. The next invocation uses the same stable W&B run ID with resume enabled.

Useful controls:

```bash
# Refuse to start unless a resumable checkpoint exists
run_experiment --experiment linformer --resume required ...

# Deliberately create a separate run instead of resuming the config-hash run
run_experiment --experiment linformer --new-run ...

# Explicitly prohibit checkpoint reuse; use a new output run to avoid mixed logs
run_experiment --experiment linformer --resume never --new-run ...

# Stop the suite on its first failing job
run_experiment --experiment linformer --fail-fast ...
```

Independent jobs continue when `--fail-fast` is not set. Failed jobs retain their traceback in `manifest.json`; dependent jobs are marked blocked. After correcting the issue, rerun the same command.

## 6. Output layout

```text
/mnt/results/longdoc/<paper>/<run-id>/
  manifest.json
  resolved_config.yaml
  suite_plan.yaml
  logs/
    run.log
    events.jsonl
  cells/<registry-cell>/
    pretrain/
      checkpoint_latest.pt
      checkpoint_last.pt
      checkpoint_tokens_*.pt
      train.jsonl
      heartbeat.json
      logs/
      wandb/
    finetune/<task>/seed_<seed>/
      checkpoint_latest.pt
      checkpoint_best.pt
      checkpoint_last.pt
      train.jsonl
      validation.jsonl
      test_metrics.json
      test_predictions.npz
      heartbeat.json
      logs/
      wandb/
    benchmark/
      metrics.json
      plots/benchmark.svg
    plots/
  curves/<cell>/<task>/tokens_<count>/
  analyses/<cell>/<ablation>/seed_<seed>/
    result.json
    plot.svg
    logs/
    wandb/
  summaries/
    cells/<cell>.json
    analyses.json
    compute_curves.json
    efficiency.json
    suite_summary.json
  plots/
    pretraining_loss.svg
    validation_curves.svg
    task_scores.svg
    accuracy_vs_pretraining_compute.svg
    throughput_by_length.svg
```

Checkpoint writes, manifests, and JSON status files use atomic replacement. Text logs rotate, JSONL events are append-only, and `.suite.lock` prevents duplicate processes from writing to the same run.

## 7. Run one low-level cell while debugging

Paper suites use `--experiment`; individual implementation cells use `--cell`:

```bash
run_experiment \
  --cell linformer.resampled_exact_cls \
  --stage all \
  --task stress_retrieve \
  --config configs/quick.yaml \
  --data-dir /mnt/datasets/longdoc \
  --output-dir /mnt/results/debug
```

Use `--set key=value` for a scoped override, for example `--set train.mlm_tokens=10000000`. The resolved config is copied into each training directory.

## 8. Aggregate completed paper suites

After multiple paper suites share one output root, create a cross-suite JSON summary with:

```bash
run_experiment --stage summarize --output-dir /mnt/results/longdoc
```

The aggregate reader understands both the paper-suite layout and older low-level cell runs. Mean rank is computed only for controlled baseline cells that have all four real-task results; incompatible raw metrics are never averaged.

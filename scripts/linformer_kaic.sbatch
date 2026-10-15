#!/bin/bash
#SBATCH --job-name=linformer
#SBATCH --partition=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/home/rajivs/tejashmore/logs/linformer_%j.out

# Runs the whole Linformer suite (29 jobs) on one GPU with the "Option B" storage plan.
# Set --partition to the partition that has the ADA6000 cards (see `sinfo`); some clusters
# also want the card type in --gres, e.g. gpu:<type>:1.
# To resume after a time limit or crash, submit this same file again without changes.

set -uo pipefail

DATA=/scratch/users/rajivs/datasets_rajivs/tejash_temp   # data only (download_data.py --data-dir $DATA)
WORK=/home/rajivs/tejashmore                             # everything else
CODE=$WORK/dlnlp
ENV=$WORK/envs/longdoc
OUT=$WORK/results          # keep DATA, OUT and RUN_ID identical for every resubmission
RUN_ID=main

# If `conda` is not found inside jobs, load it first (for example `module load anaconda3`).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
cd "$CODE"

echo "host $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) | start $(date)"

# Option B: every 10 minutes, shrink checkpoints of finished runs (unfinished runs are never touched).
( while true; do python scripts/compact_finished.py "$OUT/linformer/$RUN_ID"; sleep 600; done ) &
CLEANER=$!

run_experiment --experiment linformer --run-id "$RUN_ID" \
  --data-dir "$DATA" --output-dir "$OUT" \
  --device cuda --wandb-mode offline \
  --set train.gradient_checkpointing=false \
  --set train.num_workers=8 \
  --set runtime.keep_last_checkpoints=0 &
RUN=$!

# Slurm's early-warning signal reaches only this script, so pass it on: the experiment then
# saves a checkpoint and exits, and the next submission resumes from it.
trap 'echo "signal received $(date): asking the experiment to save a checkpoint"; kill -USR1 $RUN 2>/dev/null' USR1 TERM
wait $RUN; STATUS=$?
while kill -0 $RUN 2>/dev/null; do wait $RUN; STATUS=$?; done

kill $CLEANER 2>/dev/null
python scripts/compact_finished.py "$OUT/linformer/$RUN_ID"
echo "experiment exit code $STATUS | end $(date)"
head -8 "$OUT/linformer/$RUN_ID/status/CHECKLIST.md" 2>/dev/null
exit $STATUS

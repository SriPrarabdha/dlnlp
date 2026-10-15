#!/bin/bash
#SBATCH --job-name=linformer-check
#SBATCH --partition=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rajivs/tejashmore/logs/linformer_check_%j.out

# One-time check before the real run, so nothing has to run on the login node:
# GPU + PyTorch, the transferred data, the unit tests, and a ~10-minute end-to-end run of the
# whole Linformer suite with tiny settings (its accuracies are meaningless, near chance).
# Use the same --partition as scripts/linformer_kaic.sbatch.

set -uo pipefail

DATA=/scratch/users/rajivs/datasets_rajivs/tejash_temp
WORK=/home/rajivs/tejashmore
CODE=$WORK/dlnlp
ENV=$WORK/envs/longdoc

# If `conda` is not found inside jobs, load it first (for example `module load anaconda3`).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
cd "$CODE"

echo "== 1/4 GPU and PyTorch"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| GPU available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0), '| bf16:', torch.cuda.is_bf16_supported())"

echo "== 2/4 Transferred data"
(cd "$DATA" && sha256sum --quiet -c SHA256SUMS && echo "all $(wc -l < SHA256SUMS) data files match their checksums")
python -c "import json; m = json.load(open('$DATA/manifest.json')); print('tokenizer vocab', m['tokenizer_vocab_size']); [print(d['name'], d['splits']) for d in m['datasets']]"

echo "== 3/4 Unit tests (expect 55 passed)"
python -m pytest -q -p no:cacheprovider 2>&1 | tail -3

echo "== 4/4 End-to-end smoke test (expect Progress: 29/29)"
rm -rf "$WORK/smoke"
run_experiment --experiment linformer --run-id smoke --config configs/quick.yaml \
  --data-dir "$DATA" --output-dir "$WORK/smoke" --device cuda --no-wandb > "$WORK/logs/smoke_${SLURM_JOB_ID:-manual}.log" 2>&1
echo "smoke test exit code $? (0 = OK; full log: $WORK/logs/smoke_${SLURM_JOB_ID:-manual}.log)"
grep -E "State:|Progress:" "$WORK/smoke/linformer/smoke/status/CHECKLIST.md"
rm -rf "$WORK/smoke"
echo "== done $(date)"

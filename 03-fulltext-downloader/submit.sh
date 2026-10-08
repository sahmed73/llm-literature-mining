#!/bin/bash
#SBATCH --job-name=litmine-fulltext
#SBATCH --partition=pi.amartini
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=output/slurm-%j.out
#SBATCH --error=output/slurm-%j.err

# Settings (model, API keys, paths) from ../.env if present; values passed with
# `sbatch --export=ALL,LLM_MODEL=...` take precedence.
if [ -f ../.env ]; then set -a; source ../.env; set +a; fi

set -euo pipefail

echo "============================================"
echo "Full-text downloader"
echo "Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "Node:      ${SLURM_NODELIST:-N/A}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Started:   $(date)"
echo "============================================"

# Activate micromamba env
eval "$(micromamba shell hook -s bash)"
micromamba activate ao_mining

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p output output/final_text

# Optional overrides:
#   sbatch --export=ALL,INPUT_JSON=/path/file.json,EXTRA_ARGS="--no-resume" submit.sh
INPUT_JSON="${INPUT_JSON:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

CMD=(python -u main.py --workers "$WORKERS")
if [[ -n "$INPUT_JSON" ]]; then
  CMD+=(--input "$INPUT_JSON")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"

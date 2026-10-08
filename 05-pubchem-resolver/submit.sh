#!/bin/bash
#SBATCH --job-name=litmine-pubchem
#SBATCH --partition=pi.amartini
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --time=12:00:00

# Settings (model, API keys, paths) from ../.env if present; values passed with
# `sbatch --export=ALL,LLM_MODEL=...` take precedence.
if [ -f ../.env ]; then set -a; source ../.env; set +a; fi

echo "============================================"
echo "PubChem resolver"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Started:   $(date)"
echo "============================================"

source ~/.bashrc
micromamba activate ao_mining

cd "$SLURM_SUBMIT_DIR" || exit 1

python -u main.py

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"

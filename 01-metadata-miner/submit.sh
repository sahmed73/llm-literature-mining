#!/bin/bash
#SBATCH --job-name=litmine-metadata
#SBATCH --partition=pi.amartini
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=10-00:00:00
#SBATCH --mem=90G

# Settings (model, API keys, paths) from ../.env if present; values passed with
# `sbatch --export=ALL,LLM_MODEL=...` take precedence.
if [ -f ../.env ]; then set -a; source ../.env; set +a; fi

echo "=========================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $HOSTNAME"
echo "CPUs      : $SLURM_CPUS_ON_NODE"
echo "Started   : $(date)"
echo "=========================================="

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate ao_mining

START_TIME=$(date +%s)

python main.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
DAYS=$((ELAPSED / 86400))
HOURS=$(( (ELAPSED % 86400) / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$((ELAPSED % 60))

echo "=========================================="
echo "Finished  : $(date)"
printf "Elapsed   : %d day(s), %02d:%02d:%02d\n" $DAYS $HOURS $MINS $SECS
echo "=========================================="

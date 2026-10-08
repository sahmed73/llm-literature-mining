#!/bin/bash
#SBATCH --job-name=litmine-scorer
#SBATCH --partition=pi.amartini
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=120G

# Settings (model, API keys, paths) from ../.env if present; values passed with
# `sbatch --export=ALL,LLM_MODEL=...` take precedence.
if [ -f ../.env ]; then set -a; source ../.env; set +a; fi

echo "=========================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $HOSTNAME"
echo "GPUs      : $CUDA_VISIBLE_DEVICES"
echo "Started   : $(date)"
echo "=========================================="

# ── Activate environment ──────────────────────────────────────────
eval "$(micromamba shell hook --shell bash)"
micromamba activate ao_mining
echo "Environment activated."

# ── Create logs dir ───────────────────────────────────────────────
mkdir -p logs output

# ── Start Ollama ──────────────────────────────────────────────────
OLLAMA="${OLLAMA_BIN:-ollama}"

echo "Starting Ollama server..."
$OLLAMA serve > logs/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!

# Give Ollama time to initialize
sleep 20
echo "Ollama started (PID: $OLLAMA_PID)"

# ── Run scorer ────────────────────────────────────────────────────
START_TIME=$(date +%s)

echo "Running scorer with model ${LLM_MODEL:-llama3:8b}..."
python -u main.py --resume

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
DAYS=$((ELAPSED / 86400))
HOURS=$(( (ELAPSED % 86400) / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$((ELAPSED % 60))

# ── Cleanup ───────────────────────────────────────────────────────
echo "Stopping Ollama..."
kill $OLLAMA_PID 2>/dev/null

echo "=========================================="
echo "Finished  : $(date)"
printf "Elapsed   : %d day(s), %02d:%02d:%02d\n" $DAYS $HOURS $MINS $SECS
echo "=========================================="

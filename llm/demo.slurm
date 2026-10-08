#!/bin/bash
#SBATCH --job-name=litmine-llm-demo
#SBATCH --partition=pi.amartini
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=llm-demo-%j.out
#
# Start a private Ollama server on a GPU node and run llm/demo.py with any model.
# Submit from the repository root:
#
#   sbatch llm/demo.slurm                                                # Llama 3 8B
#   sbatch --mem=64G --export=ALL,LLM_MODEL=llama3:70b llm/demo.slurm    # Llama 3 70B
#   sbatch --export=ALL,LLM_MODEL=qwen3:30b-a3b,LLM_THINK=false,LLM_FORCE_JSON=true llm/demo.slurm
#
# GPU memory measured on a 48 GB NVIDIA L40S: Llama 3 8B 6 GB, Qwen 3 30B-A3B 21 GB,
# Llama 3 70B 41 GB (fits one GPU at its 8K context).

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
if [ -f .env ]; then set -a; source .env; set +a; fi
MODEL="${LLM_MODEL:-llama3:8b}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate "${CONDA_ENV:-ao_mining}"

# A per-job port so several jobs can share a node.
PORT=$((11500 + ${SLURM_JOB_ID:-0} % 400))
export OLLAMA_HOST="127.0.0.1:${PORT}"
"${OLLAMA_BIN:-ollama}" serve > "llm-demo-ollama-${SLURM_JOB_ID:-local}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null' EXIT

for _ in $(seq 1 120); do
    curl -s "http://${OLLAMA_HOST}/api/tags" > /dev/null && break
    sleep 1
done

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -u llm/demo.py --model "$MODEL" --url "http://${OLLAMA_HOST}"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

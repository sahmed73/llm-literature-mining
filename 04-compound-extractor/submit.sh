#!/bin/bash
#SBATCH --job-name=litmine-extractor
#SBATCH --partition=pi.amartini
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

# Settings (model, API keys, paths) from ../.env if present; values passed with
# `sbatch --export=ALL,LLM_MODEL=...` take precedence.
if [ -f ../.env ]; then set -a; source ../.env; set +a; fi

echo "============================================"
echo "Compound extractor"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Started:   $(date)"
echo "============================================"

# ── Activate environment ──────────────────────────────────────────────────────
source ~/.bashrc
micromamba activate ao_mining

cd "$SLURM_SUBMIT_DIR" || exit 1

# ── Start Ollama server ──────────────────────────────────────────────────────
OLLAMA_BIN="${OLLAMA_BIN:-ollama}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
NUM_OLLAMA_SERVERS=$(python - <<'PY'
import config as cfg
print(cfg.NUM_OLLAMA_SERVERS)
PY
)
OLLAMA_BASE_PORT=$(python - <<'PY'
import config as cfg
print(cfg.OLLAMA_BASE_PORT)
PY
)

log_gpu_status() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU status"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv,noheader,nounits || true
}

start_gpu_monitor() {
    (
        while true; do
            sleep 60
            log_gpu_status
        done
    ) &
    GPU_MONITOR_PID=$!
}

stop_gpu_monitor() {
    if [ -n "${GPU_MONITOR_PID:-}" ]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
}

OLLAMA_PIDS=()

start_ollama_servers() {
    echo "Starting Ollama servers..."
    for ((i=0; i<NUM_OLLAMA_SERVERS; i++)); do
        port=$((OLLAMA_BASE_PORT + i))
        host="0.0.0.0:${port}"
        log_path="${SLURM_SUBMIT_DIR}/ollama-${SLURM_JOB_ID}-${port}.log"
        echo "Ollama server $((i + 1)): http://localhost:${port}"
        echo "Ollama server $((i + 1)) log: ${log_path}"
        OLLAMA_HOST="$host" $OLLAMA_BIN serve > "$log_path" 2>&1 &
        OLLAMA_PIDS+=($!)
    done
}

wait_for_ollama_servers() {
    echo "Waiting for Ollama servers to start..."
    for i in $(seq 1 60); do
        ready=1
        for ((j=0; j<NUM_OLLAMA_SERVERS; j++)); do
            port=$((OLLAMA_BASE_PORT + j))
            if ! curl -s "http://localhost:${port}/api/tags" > /dev/null 2>&1; then
                ready=0
                break
            fi
        done
        if [ "$ready" -eq 1 ]; then
            echo "All ${NUM_OLLAMA_SERVERS} Ollama servers ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "Timed out waiting for Ollama servers."
    return 1
}

stop_ollama_servers() {
    for pid in "${OLLAMA_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${OLLAMA_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}

echo "Starting Ollama servers..."
echo "Configured Ollama servers: $NUM_OLLAMA_SERVERS"
echo "Base port: $OLLAMA_BASE_PORT"
log_gpu_status
start_ollama_servers
trap 'stop_gpu_monitor; stop_ollama_servers' EXIT

# Wait for Ollama servers to be ready
wait_for_ollama_servers || exit 1

log_gpu_status
start_gpu_monitor

# ── Run pipeline ─────────────────────────────────────────────────────────────
python -u main.py

stop_gpu_monitor
log_gpu_status

# ── Cleanup ──────────────────────────────────────────────────────────────────
echo "Stopping Ollama servers..."
stop_ollama_servers

log_gpu_status

echo ""
echo "============================================"
echo "Finished: $(date)"
echo "============================================"

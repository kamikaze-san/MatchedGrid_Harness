#!/bin/bash
# Slurm script for one cell of the matched grid.
# Disconnect-safe (Slurm detached), resume-safe (runner.py reads done.txt),
# timeout-safe (SIGUSR1 5min before --time), failure-safe (--requeue).
#
# Submit:
#   sbatch --export=ALL,CELL_CFG=configs/cells/G1_<id>.json sbatch/run_cell.sbatch
# Partition, QoS and GPU type below are those of our cluster; edit them for yours.
#SBATCH --job-name=strata_cell
#SBATCH --partition=gpu-1day
#SBATCH --qos=gpu-1day
#SBATCH --gres=gpu:a100-80gb:2
#SBATCH --time=23:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=logs/cell_%x_%j.out
#SBATCH --error=logs/cell_%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -e
set -o pipefail
mkdir -p logs

echo "=== Cell job started ==="
echo "Job:          $SLURM_JOB_ID  on  $(hostname)"
echo "GPU visible:  $CUDA_VISIBLE_DEVICES"
IFS=, read -r GPU_A GPU_B <<< "$CUDA_VISIBLE_DEVICES"; GPU_B=${GPU_B:-$GPU_A}
echo "Cell:         ${CELL_CFG:-(unset)}"
echo "Start:        $(date)"

if [[ -z "${CELL_CFG:-}" ]]; then
    echo "ERROR: CELL_CFG not set" >&2
    exit 2
fi

# --- environment: adjust for your cluster -------------------------------------
# e.g. MODULES="cuda/12.8" CONDA_SH=/path/to/miniconda/etc/profile.d/conda.sh
if [[ -n "${MODULES:-}" ]]; then module purge; module load $MODULES; fi
if [[ -n "${CONDA_SH:-}" ]]; then source "$CONDA_SH"; fi
conda activate "${CONDA_ENV:-strata}"

export HF_HOME=${HF_HOME:-$HOME/scratch/hf_cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$HOME/scratch/vllm_cache}
export HF_HUB_DISABLE_XET=1
export HF_TOKEN=${HF_TOKEN:-$(cat $HOME/.hf_token 2>/dev/null)}
export HUGGING_FACE_HUB_TOKEN=""
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
# LLaVA-OneVision and some other VLMs report max_position_embeddings=2048 in their
# config.json even though they can serve longer contexts via RoPE scaling. We use
# 4096 sequences (well within reach for 6-turn dialogues) so override is safe.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

CFG="$CELL_CFG"
TARGET_MODEL=$(python -c "import json; print(json.load(open('$CFG'))['target'])")
ATTACKER_MODEL=$(python -c "import json; print(json.load(open('$CFG'))['attacker'])")
JUDGE_MODEL=$(python -c "import json; print(json.load(open('$CFG'))['judge'])")
TARGET_PORT=$(python -c "import json; print(json.load(open('$CFG'))['target_port'])")
ATTACKER_PORT=$(python -c "import json; print(json.load(open('$CFG'))['attacker_port'])")
JUDGE_PORT=$(python -c "import json; print(json.load(open('$CFG'))['judge_port'])")
CELL_ID=$(python -c "import json; print(json.load(open('$CFG'))['cell_id'])")

echo "Target:    $TARGET_MODEL :: $TARGET_PORT (GPU 0)"
echo "Attacker:  $ATTACKER_MODEL :: $ATTACKER_PORT (GPU 1)"
echo "Judge:     $JUDGE_MODEL :: $JUDGE_PORT"

LOGDIR="logs/$CELL_ID"
mkdir -p "$LOGDIR"

TARGET_EXTRA_ARGS=""
if [[ "$TARGET_MODEL" == *"Pixtral"* ]]; then
    TARGET_EXTRA_ARGS="--tokenizer_mode mistral --config_format mistral --load_format mistral"
fi

# Server 1: target (multimodal) on GPU 0
echo "Launching target vLLM on GPU 0..."
CUDA_VISIBLE_DEVICES=$GPU_A python -m vllm.entrypoints.openai.api_server \
    --model "$TARGET_MODEL" --port "$TARGET_PORT" --host 127.0.0.1 \
    --download-dir "$HF_HOME" --max-model-len ${TARGET_MAX_LEN:-4096} \
    --gpu-memory-utilization ${TARGET_GPU_UTIL:-0.85} --dtype auto --trust-remote-code \
    --limit-mm-per-prompt image=${MM_LIMIT:-2} --disable-log-requests $TARGET_EXTRA_ARGS \
    > "$LOGDIR/target.log" 2>&1 &
TGT_PID=$!

# Single-GPU mode (job allocated 1 GPU -> GPU_A == GPU_B): vLLM counts other
# processes' memory against its own budget, so start the attacker only after the
# target has claimed its share. Use a low TARGET_GPU_UTIL and a high ATTACKER_GPU_UTIL.
if [[ "$GPU_A" == "$GPU_B" ]]; then
    echo "Single-GPU mode: waiting for target before launching attacker on GPU $GPU_A..."
    for i in $(seq 1 360); do
        curl -sf "http://127.0.0.1:${TARGET_PORT}/v1/models" > /dev/null 2>&1 && break
        kill -0 $TGT_PID 2>/dev/null || { echo "FATAL: target died during startup" >&2; tail -25 "$LOGDIR/target.log" >&2; exit 3; }
        sleep 5
    done
fi

# Server 2: attacker+judge on GPU 1 (shared instance if same port)
if [[ "$ATTACKER_PORT" == "$JUDGE_PORT" ]]; then
    echo "Launching attacker+judge (shared) vLLM on GPU 1..."
    CUDA_VISIBLE_DEVICES=$GPU_B python -m vllm.entrypoints.openai.api_server \
        --model "$ATTACKER_MODEL" --port "$ATTACKER_PORT" --host 127.0.0.1 \
        --download-dir "$HF_HOME" --max-model-len 4096 \
        --gpu-memory-utilization ${ATTACKER_GPU_UTIL:-0.85} --dtype auto --trust-remote-code --disable-log-requests \
        > "$LOGDIR/attacker_judge.log" 2>&1 &
    ATT_PID=$!
    JDG_PID=""
else
    # Fallback: separate attacker and judge servers (used when user has a real
    # safety-specialized judge model like Llama-Guard-3 on a different port).
    CUDA_VISIBLE_DEVICES=$GPU_B python -m vllm.entrypoints.openai.api_server \
        --model "$ATTACKER_MODEL" --port "$ATTACKER_PORT" --host 127.0.0.1 \
        --download-dir "$HF_HOME" --max-model-len 4096 \
        --gpu-memory-utilization 0.45 --dtype auto --trust-remote-code --disable-log-requests \
        > "$LOGDIR/attacker.log" 2>&1 &
    ATT_PID=$!
    CUDA_VISIBLE_DEVICES=$GPU_B python -m vllm.entrypoints.openai.api_server \
        --model "$JUDGE_MODEL" --port "$JUDGE_PORT" --host 127.0.0.1 \
        --download-dir "$HF_HOME" --max-model-len 4096 \
        --gpu-memory-utilization 0.35 --dtype auto --trust-remote-code --disable-log-requests \
        > "$LOGDIR/judge.log" 2>&1 &
    JDG_PID=$!
fi
echo "Server PIDs: target=$TGT_PID attacker=$ATT_PID judge=${JDG_PID:-shared}"

wait_for_port() {
    local port=$1; local name=$2; local timeout=${3:-1500}; local elapsed=0
    while ! curl -sf "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; do
        if grep -qE "No available memory|larger than the maximum number of tokens|out of memory|OutOfMemoryError|No supported config format|ValueError:|Traceback \(most recent" "$LOGDIR/${name}.log" 2>/dev/null; then
            echo "FATAL: $name died during startup -- aborting early (after ${elapsed}s)" >&2
            tail -25 "$LOGDIR/${name}.log" >&2
            kill $TGT_PID $ATK_PID $JDG_PID 2>/dev/null
            exit 3
        fi
        sleep 5; elapsed=$((elapsed + 5))
        if [ $elapsed -ge $timeout ]; then
            echo "ERROR: $name on port $port did not start within ${timeout}s" >&2
            tail -50 "$LOGDIR/${name}.log" >&2 2>/dev/null || true
            kill $TGT_PID $ATT_PID ${JDG_PID:-} 2>/dev/null || true
            exit 3
        fi
    done
    echo "$name ready on port $port (after ${elapsed}s)"
}

wait_for_port "$TARGET_PORT"   "target"
wait_for_port "$ATTACKER_PORT" "attacker_judge"
if [[ -n "${JDG_PID:-}" && "$ATTACKER_PORT" != "$JUDGE_PORT" ]]; then
    wait_for_port "$JUDGE_PORT" "judge"
fi

RUNNER_PID=""
cleanup_handler() {
    echo "[sbatch] SIGUSR1 received — forwarding to runner $RUNNER_PID"
    [[ -n "$RUNNER_PID" ]] && kill -USR1 "$RUNNER_PID" 2>/dev/null || true
    for i in $(seq 1 240); do
        kill -0 "$RUNNER_PID" 2>/dev/null || break
        sleep 1
    done
    kill $TGT_PID $ATT_PID ${JDG_PID:-} 2>/dev/null || true
    echo "[sbatch] Clean shutdown."
    exit 0
}
trap cleanup_handler USR1

echo "Starting runner..."
python -m src.runner --cell-config "$CFG" &
RUNNER_PID=$!
wait $RUNNER_PID
RUNNER_EXIT=$?

echo "Runner exited $RUNNER_EXIT. Stopping vLLM servers..."
kill $TGT_PID $ATT_PID ${JDG_PID:-} 2>/dev/null || true
wait $TGT_PID 2>/dev/null || true
wait $ATT_PID 2>/dev/null || true
[[ -n "${JDG_PID:-}" ]] && wait $JDG_PID 2>/dev/null || true

echo "End: $(date)"
exit $RUNNER_EXIT

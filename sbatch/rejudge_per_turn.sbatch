#!/bin/bash
# Offline PER-TURN re-judge sbatch. Scores every turn of every saved trajectory,
# not just the final one -- no new generation, pure re-read of trajectories.jsonl.
# Submit with:
#   sbatch --export=ALL,JUDGE_MODEL=Qwen/Qwen2.5-7B-Instruct,JUDGE_SUFFIX=qwen7b_perturn,CELL_FILTER=crescendo sbatch/rejudge_per_turn.sbatch
# Partition, QoS and GPU type below are those of our cluster; edit them for yours.
#SBATCH --job-name=strata_rejudge_perturn
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --time=07:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=logs/rejudge_perturn_%x_%j.out
#SBATCH --error=logs/rejudge_perturn_%x_%j.err

set -e
set -o pipefail
mkdir -p logs

echo "=== rejudge_per_turn started ==="
echo "Job: $SLURM_JOB_ID  on $(hostname)"
echo "Judge: $JUDGE_MODEL"
echo "Suffix: $JUDGE_SUFFIX"
echo "Cell filter: ${CELL_FILTER:-(none -- ALL cells, expensive)}"
echo "Start: $(date)"

if [[ -z "${JUDGE_MODEL:-}" || -z "${JUDGE_SUFFIX:-}" ]]; then
    echo "ERROR: JUDGE_MODEL and JUDGE_SUFFIX env vars required" >&2
    exit 2
fi

# --- environment: adjust for your cluster -------------------------------------
# e.g. MODULES="cuda/12.8" CONDA_SH=/path/to/miniconda/etc/profile.d/conda.sh
if [[ -n "${MODULES:-}" ]]; then module purge; module load $MODULES; fi
if [[ -n "${CONDA_SH:-}" ]]; then source "$CONDA_SH"; fi
conda activate "${CONDA_ENV:-strata}"

export HF_HOME=${HF_HOME:-$HOME/scratch/hf_cache}
export HF_TOKEN=${HF_TOKEN:-$(cat $HOME/.hf_token 2>/dev/null)}
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HF_HUB_DISABLE_XET=1

JUDGE_PORT=${JUDGE_PORT:-8091}
LOGDIR="logs/rejudge_perturn_${JUDGE_SUFFIX}_${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

EXTRA_ARGS=""
if [[ "$JUDGE_MODEL" == *"72B"* ]] || [[ "$JUDGE_MODEL" == *"AWQ"* ]]; then
    EXTRA_ARGS="--quantization awq"
fi

# vLLM hands --download-dir to huggingface_hub as cache_dir, so it resolves
# $HF_HOME/models--org--repo/ (top level) -- NOT the standard $HF_HOME/hub/...
# that snapshot_download() writes to. When those two diverge, online mode silently
# re-fetches over the network (dies on this cluster's flaky compute-node DNS) and
# HF_HUB_OFFLINE=1 dies with "No supported config format found". Setting
# JUDGE_LOCAL_PATH to a snapshot dir skips hub resolution entirely: no network,
# no cache-layout guessing. --served-model-name keeps the API name canonical,
# which src/judge.py relies on for its Llama-Guard routing.
MODEL_ARG="$JUDGE_MODEL"
PATH_ARGS="--download-dir $HF_HOME"
if [[ -n "${JUDGE_LOCAL_PATH:-}" ]]; then
    if [[ ! -f "$JUDGE_LOCAL_PATH/config.json" ]]; then
        echo "ERROR: JUDGE_LOCAL_PATH has no config.json: $JUDGE_LOCAL_PATH" >&2
        exit 2
    fi
    MODEL_ARG="$JUDGE_LOCAL_PATH"
    PATH_ARGS="--served-model-name $JUDGE_MODEL"
    echo "Loading judge from local path: $JUDGE_LOCAL_PATH (served as $JUDGE_MODEL)"
fi

echo "Launching judge vLLM..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ARG" --port "$JUDGE_PORT" --host 127.0.0.1 \
    $PATH_ARGS --max-model-len 4096 \
    --gpu-memory-utilization ${JUDGE_GPU_UTIL:-0.5} --dtype auto \
    --trust-remote-code --disable-log-requests $EXTRA_ARGS \
    > "$LOGDIR/judge.log" 2>&1 &
JDG_PID=$!

wait_for_port() {
    local port=$1; local timeout=${2:-1800}; local elapsed=0
    while ! curl -sf "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; do
        kill -0 $JDG_PID 2>/dev/null || { echo "ERROR: judge vLLM died" >&2; tail -30 "$LOGDIR/judge.log" >&2; exit 3; }
        sleep 5; elapsed=$((elapsed + 5))
        if [ $elapsed -ge $timeout ]; then
            echo "ERROR: judge did not start within ${timeout}s" >&2
            tail -50 "$LOGDIR/judge.log" >&2 2>/dev/null
            kill $JDG_PID 2>/dev/null
            exit 3
        fi
    done
    echo "judge HTTP server up on port $port (after ${elapsed}s) -- NOT a readiness guarantee"
}

# Same fix as sbatch/rejudge.sbatch: /v1/models answers before the engine finishes
# loading, so probe a real chat completion until it genuinely succeeds.
wait_for_chat_ready() {
    local port=$1; local timeout=${2:-1800}; local elapsed=0
    local probe_payload
    probe_payload=$(python3 -c "import json,os; print(json.dumps({'model': os.environ['JUDGE_MODEL'], 'messages': [{'role':'user','content':'ping'}], 'max_tokens': 1}))")
    while true; do
        if curl -sf -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
             -H "Content-Type: application/json" -d "$probe_payload" > /dev/null 2>&1; then
            echo "judge engine actually serving real chat completions on port $port (after ${elapsed}s)"
            return 0
        fi
        sleep 10; elapsed=$((elapsed + 10))
        if [ $elapsed -ge $timeout ]; then
            echo "ERROR: judge never served a real chat completion within ${timeout}s" >&2
            tail -80 "$LOGDIR/judge.log" >&2 2>/dev/null
            kill $JDG_PID 2>/dev/null
            exit 3
        fi
    done
}
wait_for_port "$JUDGE_PORT" 1800
wait_for_chat_ready "$JUDGE_PORT" 1800

echo "Running per-turn rejudge..."
echo "Cell glob: ${CELL_GLOB:-G1_* (default -- main grid only, NOT CRESfix_*/E3fix_*)}"
python -m src.rejudge_per_turn \
    --judge-name "$JUDGE_MODEL" \
    --judge-port "$JUDGE_PORT" \
    --results-dir results \
    --output-suffix "$JUDGE_SUFFIX" \
    ${CELL_GLOB:+--cell-glob "$CELL_GLOB"} \
    ${CELL_FILTER:+--cell-filter "$CELL_FILTER"}

REJUDGE_EXIT=$?

echo "Rejudge_per_turn exited $REJUDGE_EXIT. Stopping vLLM..."
kill $JDG_PID 2>/dev/null || true
wait $JDG_PID 2>/dev/null || true

echo "End: $(date)"
exit $REJUDGE_EXIT

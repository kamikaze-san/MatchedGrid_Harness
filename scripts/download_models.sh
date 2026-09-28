#!/bin/bash
# Download every model used in the paper into $HF_HOME (run once, on a node with internet).
# Llama-3.1-8B-Instruct and Llama-Guard-3-8B are gated: accept their licences on huggingface.co first.
set -euo pipefail
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
mkdir -p "$HF_HOME"
MODELS=(
    "OpenGVLab/InternVL2_5-26B"
    "OpenGVLab/InternVL2_5-8B"
    "Qwen/Qwen2.5-72B-Instruct-AWQ"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-VL-32B-Instruct"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-Guard-3-8B"
    "microsoft/Phi-3.5-vision-instruct"
    "mistralai/Pixtral-12B-2409"
    "openbmb/MiniCPM-V-2_6"
)
for m in "${MODELS[@]}"; do
    echo "==> $m"
    huggingface-cli download "$m" --cache-dir "$HF_HOME"
done

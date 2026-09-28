#!/bin/bash
# Create the conda environment used for all runs (Python 3.11, vLLM 0.7.3, transformers 4.49).
set -euo pipefail
ENV_NAME=${ENV_NAME:-strata}
conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"
pip install -r requirements.txt
python -c "import vllm, transformers; print('vllm', vllm.__version__, '| transformers', transformers.__version__)"

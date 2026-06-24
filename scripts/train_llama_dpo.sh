#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-${DIARC_MODEL_DIR:-models}/Llama-3.2-3B-ReArc-merged}"

python3 -m diarc.train_dpo_llama \
  --dataset-subdir "${DPO_DATASET_SUBDIR:-dpo_conceptarc_morphology}" \
  --output-subdir "${DPO_OUTPUT_SUBDIR:-llama3b-dpo-conceptarc-morphology}"

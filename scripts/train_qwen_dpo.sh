#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python3 -m diarc.train_dpo_qwen \
  --base-model-path "${BASE_MODEL_PATH:-${DIARC_MODEL_DIR:-models}/qwen3_4b_arc_sft}" \
  --dataset-subdir "${DPO_DATASET_SUBDIR:-dpo_conceptarc_morphology}" \
  --output-subdir "${DPO_OUTPUT_SUBDIR:-qwen3-dpo-conceptarc-morphology}"

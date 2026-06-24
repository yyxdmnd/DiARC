#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-${DIARC_MODEL_DIR:-models}/Mistral-NeMo-Minitron-8B-ARC-SFT}"

python3 -m diarc.train_dpo_minitron \
  --dataset-subdir "${DPO_DATASET_SUBDIR:-dpo_conceptarc_morphology}" \
  --output-subdir "${DPO_OUTPUT_SUBDIR:-minitron-dpo-conceptarc-morphology}"

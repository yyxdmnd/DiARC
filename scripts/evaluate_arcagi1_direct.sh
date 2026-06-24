#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python3 -m diarc.evaluate_arcagi1_cli \
  --mode "${EVAL_MODE:-direct}" \
  --output-subdir "${EVAL_OUTPUT_SUBDIR:-arcagi1-direct}" \
  --input-aug-n "${INPUT_AUG_N:-1}" \
  --num-return-sequences "${NUM_RETURN_SEQUENCES:-2}"

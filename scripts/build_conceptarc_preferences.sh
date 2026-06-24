#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python3 -m diarc.build_external_preferences \
  --dataset conceptarc \
  --dataset-root "${DIARC_DATA_DIR:-data}/ConceptARC" \
  --output-dir "${DIARC_DATA_DIR:-data}/dpo_conceptarc_morphology" \
  --transform-category morphology \
  --top-k 16 \
  --ranker auto

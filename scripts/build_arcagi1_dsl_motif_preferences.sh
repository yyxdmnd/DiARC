#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python3 -m diarc.build_dsl_motif_preferences \
  --rearc-root "${DIARC_RE_ARC_GEN_DIR:-re_arc_gen}" \
  --output "${DIARC_DATA_DIR:-data}/dpo_arcagi1_dsl_motif/arc_dpo_data_all.jsonl" \
  --rewrite all \
  --groups-per-task 2

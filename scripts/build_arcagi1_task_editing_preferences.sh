#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "" ]]; then
  echo "usage: $0 path/to/edited_generators_or_verifiers.py" >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python3 -m diarc.build_task_editing_preferences \
  --rearc-root "${DIARC_RE_ARC_GEN_DIR:-re_arc_gen}" \
  --edited-generator-module "$1" \
  --output "${DIARC_DATA_DIR:-data}/dpo_arcagi1_task_editing/arc_dpo_data_all.jsonl"

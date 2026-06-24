# Artifact Notes

This repository contains code for constructing ARC preference data, training
DPO adapters, and running ARC-style evaluations. It includes the small ARC-style
benchmark files used by the paper. It intentionally does not include model
checkpoints, full RE-ARC generated corpora, generated DPO preference datasets,
or raw experiment logs.

Included benchmark data:

- `data/ARC-AGI-1/`: ARC-AGI-1 public training/evaluation JSON files.
- `data/ARC-AGI-2/`: ARC-AGI-2 public training/evaluation JSON files.
- `data/Mini-ARC/`
- `data/ConceptARC/`
- `data/1D-ARC/`
- `data/arc-community/`

Expected external assets:

- RE-ARC generator resources under `re_arc_gen/` when using rule-level or
  generator-based data construction. The rule-level builders expect the usual
  `dsl.py`, `generators.py`, and `verifiers.py` files.
- Task-specific editing experiments require an edited verifier or generator
  module supplied by the user, for example a file with `verify_<task_id>` or
  `generate_<task_id>` functions.
- Local base model checkpoints under `models/`, or paths supplied through
  `BASE_MODEL_PATH` / `DIARC_MODEL_DIR`.
- Generated preference JSONL files under `data/<dpo_dataset_subdir>/`.
- Trained DPO adapter directories and evaluation artifacts under `outputs/`.

Expected local layout:

```text
DiARC/
├── data/
│   ├── ARC-AGI-1/
│   ├── ConceptARC/
│   └── dpo_conceptarc_morphology/
│       └── arc_dpo_data_all.jsonl
├── models/
│   └── <local-base-checkpoint>/
├── outputs/
│   └── <trained-adapter-or-eval-output>/
└── re_arc_gen/
    ├── dsl.py
    ├── generators.py
    └── verifiers.py
```

The original experiments used offline local checkpoints. The release code keeps
that behavior by default and lets users override all paths with environment
variables.

No public model-hosting links or private download links are required by the
code. If adapters are released separately, place them under `outputs/` or pass
their local path through `LORA_ADAPTER_PATH` during evaluation.

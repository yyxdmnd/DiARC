# Environment Setup and Running Guide

This project is designed to run from a local clone with benchmark data in
`data/`, optional base checkpoints in `models/`, generated artifacts in
`outputs/`, and optional RE-ARC generator resources in `re_arc_gen/`.

## Recommended Python Environment

The released code was smoke-tested on Linux with NVIDIA L40 GPUs and this
software stack:

```text
python==3.12
torch==2.8.0
transformers==4.52.4
trl==0.9.6
peft==0.15.2
datasets==3.6.0
accelerate==1.7.0
bitsandbytes==0.48.1
unsloth==2025.10.3
numpy==1.26.4
scipy==1.16.2
scikit-learn==1.7.2
pillow==11.3.0
```

Create an environment and install dependencies:

```bash
conda create -n diarc python=3.12 -y
conda activate diarc
python -m pip install -U pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

If your CUDA or driver setup requires a specific PyTorch wheel, install that
wheel first from the official PyTorch index, then install the remaining
requirements.

The repository has no package installation step. Keep `PYTHONPATH` pointed at
`src/` when running module commands or shell scripts.

## GPU Check

Check available GPUs:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

Run a small CUDA check on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available(), torch.cuda.device_count())
x = torch.ones((1024, 1024), device="cuda")
print(float(x.sum()))
PY
```

## Local Paths

The code uses repo-relative defaults:

```text
data/       benchmark files and generated preference JSONL files
models/     local base checkpoints
outputs/    trained adapters and evaluation outputs
re_arc_gen/ optional RE-ARC DSL/generator/verifier resources
```

Override them with environment variables:

```bash
export DIARC_DATA_DIR=/path/to/data
export DIARC_MODEL_DIR=/path/to/models
export DIARC_OUTPUT_DIR=/path/to/outputs
export DIARC_RE_ARC_GEN_DIR=/path/to/re_arc_gen
```

The training scripts run in offline mode by default and expect local model
checkpoints. You can either place checkpoints under `models/` or pass
`BASE_MODEL_PATH` / `--base-model-path`.

## Required External Assets

The repository includes small benchmark JSON files for ARC-AGI-1, ARC-AGI-2,
MiniARC, ConceptARC, 1D-ARC, and ARCcommunity.

It does not include:

- Third-party base model weights.
- Full RE-ARC generated corpora.
- Generated preference JSONL files.
- Trained LoRA adapters.
- Paper experiment outputs.

For ARC-AGI-1 rule-level construction, `re_arc_gen/` should contain compatible
`dsl.py`, `generators.py`, and `verifiers.py` files. Task-specific editing also
requires a user-provided edited verifier or generator module.

## Smoke Tests

Run syntax and import checks:

```bash
python -m compileall -q src
python - <<'PY'
import importlib
for name in [
    "diarc.arc_loader",
    "diarc.negative_transforms",
    "diarc.build_external_preferences",
    "diarc.train_dpo_llama",
    "diarc.train_dpo_minitron",
    "diarc.train_dpo_qwen",
]:
    importlib.import_module(name)
    print("OK", name)
PY
```

Check the command-line help for the main runnable entry points:

```bash
python -m diarc.build_external_preferences --help
python -m diarc.train_dpo_llama --help
python -m diarc.train_dpo_minitron --help
python -m diarc.train_dpo_qwen --help
python -m diarc.evaluate_arcagi1_cli --help
```

Build a tiny preference dataset from included files:

```bash
CUDA_VISIBLE_DEVICES=0 python -m diarc.build_external_preferences \
  --dataset miniarc \
  --dataset-root data/Mini-ARC \
  --output-dir /tmp/diarc_smoke/dpo_miniarc_grid_block \
  --transform-category grid_block \
  --top-k 2 \
  --max-tasks 2 \
  --no-augment \
  --ranker grid
```

For 1D-ARC, use the same builder with 1D-safe transforms:

```bash
CUDA_VISIBLE_DEVICES=0 python -m diarc.build_external_preferences \
  --dataset 1d-arc \
  --dataset-root data/1D-ARC \
  --output-dir /tmp/diarc_smoke/dpo_1d_arc_random \
  --transform-category random_perturb \
  --top-k 2 \
  --max-tasks 2 \
  --no-augment \
  --ranker grid
```

The smoke-test output directory can be deleted after the check:

```bash
rm -rf /tmp/diarc_smoke
```

## Preference Data Construction

The output-level builder works directly on ARC-like benchmark files:

```bash
python -m diarc.build_external_preferences \
  --dataset conceptarc \
  --dataset-root data/ConceptARC \
  --output-dir data/dpo_conceptarc_morphology \
  --transform-category morphology \
  --top-k 16 \
  --ranker auto
```

For 1D-ARC, use 1D-safe transformations:

```bash
python -m diarc.build_external_preferences \
  --dataset 1d-arc \
  --dataset-root data/1D-ARC \
  --output-dir data/dpo_1d_arc_random \
  --transform-category random_perturb \
  --top-k 16 \
  --ranker grid
```

The ARC-AGI-1 rule-level builders require compatible RE-ARC resources under
`re_arc_gen/`:

```bash
bash scripts/build_arcagi1_dsl_motif_preferences.sh
bash scripts/build_arcagi1_task_editing_preferences.sh path/to/edited_programs.py
```

Every training dataset should contain:

```text
data/<dpo_dataset_subdir>/arc_dpo_data_all.jsonl
```

## Training

Training expects `arc_dpo_data_all.jsonl` under
`$DIARC_DATA_DIR/<dataset_subdir>/`.

Llama-style checkpoints:

```bash
BASE_MODEL_PATH=models/Llama-3.2-3B-ReArc-merged \
CUDA_VISIBLE_DEVICES=0 python -m diarc.train_dpo_llama \
  --dataset-subdir dpo_conceptarc_morphology \
  --output-subdir llama3b-dpo-conceptarc-morphology
```

or with the wrapper:

```bash
DPO_DATASET_SUBDIR=dpo_conceptarc_morphology \
DPO_OUTPUT_SUBDIR=llama3b-dpo-conceptarc-morphology \
BASE_MODEL_PATH=models/Llama-3.2-3B-ReArc-merged \
CUDA_VISIBLE_DEVICES=0 bash scripts/train_llama_dpo.sh
```

Minitron checkpoints:

```bash
BASE_MODEL_PATH=models/Mistral-NeMo-Minitron-8B-ARChitects-ReArc1200-bnb-4bit \
CUDA_VISIBLE_DEVICES=0 python -m diarc.train_dpo_minitron \
  --dataset-subdir dpo_conceptarc_morphology \
  --output-subdir minitron-dpo-conceptarc-morphology
```

or with the wrapper:

```bash
DPO_DATASET_SUBDIR=dpo_conceptarc_morphology \
DPO_OUTPUT_SUBDIR=minitron-dpo-conceptarc-morphology \
BASE_MODEL_PATH=models/Minitron-8B-ARC-SFT \
CUDA_VISIBLE_DEVICES=0 bash scripts/train_minitron_dpo.sh
```

Qwen checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 python -m diarc.train_dpo_qwen \
  --base-model-path models/qwen3_4b_grids15_sft139_bfloat16 \
  --dataset-subdir dpo_conceptarc_morphology \
  --output-subdir qwen3-dpo-conceptarc-morphology
```

or with the wrapper:

```bash
DPO_DATASET_SUBDIR=dpo_conceptarc_morphology \
DPO_OUTPUT_SUBDIR=qwen3-dpo-conceptarc-morphology \
BASE_MODEL_PATH=models/qwen3_4b_arc_sft \
CUDA_VISIBLE_DEVICES=0 bash scripts/train_qwen_dpo.sh
```

For multi-GPU runs, launch the same training module with `torchrun` or
`accelerate launch` and expose the desired devices through
`CUDA_VISIBLE_DEVICES`.

## Evaluation

The ARC-AGI-1 evaluation wrapper uses environment variables for paths:

```bash
BASE_MODEL_PATH=models/Llama-3.2-3B-ReArc-merged \
ARC_DATA_PATH=data/ARC-AGI-1 \
EVAL_OUTPUT_PATH=outputs/arcagi1-direct \
CUDA_VISIBLE_DEVICES=0 python -m diarc.evaluate_arcagi1_cli \
  --mode direct \
  --input-aug-n 1 \
  --num-return-sequences 2
```

To evaluate a local LoRA adapter:

```bash
BASE_MODEL_PATH=models/Llama-3.2-3B-ReArc-merged \
LORA_ADAPTER_PATH=outputs/llama3b-dpo-conceptarc-morphology \
ARC_DATA_PATH=data/ARC-AGI-1 \
EVAL_OUTPUT_PATH=outputs/arcagi1-direct-adapter \
CUDA_VISIBLE_DEVICES=0 python -m diarc.evaluate_arcagi1_cli \
  --mode direct \
  --input-aug-n 1 \
  --num-return-sequences 2
```

For a short functionality check, set `EVAL_TASK_LIMIT=1`. Full evaluation uses
the complete ARC-AGI-1 public evaluation set.

## Adapters

The training scripts save PEFT/LoRA adapters under `outputs/`. For evaluation or
continued training, combine an adapter directory with the matching local base
checkpoint.

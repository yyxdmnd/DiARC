import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path

# Force offline mode for local training.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_real_find_spec = importlib.util.find_spec


def _patched_find_spec(name, package=None):
    if name == "torchao" or name.startswith("torchao."):
        return None
    return _real_find_spec(name, package)


importlib.util.find_spec = _patched_find_spec

try:
    from .paths import DATA_DIR, MODEL_DIR, OUTPUT_DIR
except ImportError:  # pragma: no cover
    from paths import DATA_DIR, MODEL_DIR, OUTPUT_DIR


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"environment variable {name} must be an integer, got: {value}") from exc


def load_local_tokenizer(model_path: str):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except AttributeError as exc:
        if "'dict' object has no attribute 'model_type'" not in str(exc):
            raise

    model_dir = Path(model_path)
    tokenizer_file = model_dir / "tokenizer.json"
    tokenizer_config_file = model_dir / "tokenizer_config.json"
    special_tokens_file = model_dir / "special_tokens_map.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"missing tokenizer file: {tokenizer_file}")

    tokenizer_config = {}
    special_tokens = {}
    if tokenizer_config_file.exists():
        tokenizer_config = json.loads(tokenizer_config_file.read_text())
    if special_tokens_file.exists():
        special_tokens = json.loads(special_tokens_file.read_text())

    kwargs = {}
    for key in ["bos_token", "eos_token", "unk_token", "pad_token", "cls_token", "sep_token", "mask_token"]:
        value = special_tokens.get(key, tokenizer_config.get(key))
        if isinstance(value, dict) and "content" in value:
            value = value["content"]
        if value is not None:
            kwargs[key] = value

    for key in ["model_max_length", "padding_side", "truncation_side", "clean_up_tokenization_spaces"]:
        value = tokenizer_config.get(key)
        if value is not None:
            kwargs[key] = value

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file), **kwargs)
    chat_template = tokenizer_config.get("chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def load_training_dependencies():
    try:
        import torch
        from datasets import Dataset
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer
        if not hasattr(transformers, "BloomPreTrainedModel"):
            class BloomPreTrainedModel:  # pragma: no cover - compatibility shim
                pass
            transformers.BloomPreTrainedModel = BloomPreTrainedModel
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from trl import DPOConfig, DPOTrainer
    except (ModuleNotFoundError, ImportError) as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise ImportError(
            f"training dependency import failed: {missing}\n"
            f"please verify this environment has compatible versions of:\n"
            f"  torch datasets transformers peft trl bitsandbytes accelerate"
        ) from exc

    _original_dpo_log = DPOTrainer.log
    _original_dpo_get_batch_samples = DPOTrainer.get_batch_samples
    _original_dpo_compute_loss = DPOTrainer.compute_loss

    def _patched_dpo_log(self, logs, start_time=None):
        return _original_dpo_log(self, logs)

    def _patched_dpo_get_batch_samples(self, *args, **kwargs):
        if len(args) == 3:
            return Trainer.get_batch_samples(self, *args, **kwargs)
        return _original_dpo_get_batch_samples(self, *args, **kwargs)

    def _patched_dpo_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return _original_dpo_compute_loss(self, model, inputs, return_outputs=return_outputs)

    DPOTrainer.log = _patched_dpo_log
    DPOTrainer.get_batch_samples = _patched_dpo_get_batch_samples
    DPOTrainer.compute_loss = _patched_dpo_compute_loss
    return {
        "torch": torch,
        "Dataset": Dataset,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "DPOTrainer": DPOTrainer,
        "DPOConfig": DPOConfig,
    }


def load_dpo_dataset(path: Path):
    deps = load_training_dependencies()
    Dataset = deps["Dataset"]
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(
                {
                    "prompt": item["instruction"] + item["input"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                }
            )
    return Dataset.from_list(data)


def run_training(
    negative_name: str,
    task_limit: int,
    dataset_subdir: str,
    output_subdir: str,
):
    deps = load_training_dependencies()
    torch = deps["torch"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    BitsAndBytesConfig = deps["BitsAndBytesConfig"]
    LoraConfig = deps["LoraConfig"]
    PeftModel = deps["PeftModel"]
    get_peft_model = deps["get_peft_model"]
    prepare_model_for_kbit_training = deps["prepare_model_for_kbit_training"]
    DPOTrainer = deps["DPOTrainer"]
    DPOConfig = deps["DPOConfig"]
    world_size = _get_int_env("WORLD_SIZE", 1)
    rank = _get_int_env("RANK", 0)
    local_rank = _get_int_env("LOCAL_RANK", 0)
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="从checkpoint恢复训练，可指定路径或设置为'True'自动查找最新checkpoint",
    )
    args, _ = parser.parse_known_args()

    model_name_or_path = os.environ.get(
        "BASE_MODEL_PATH",
        str(MODEL_DIR / "Llama-3.2-3B-ARChitects-ReArc-bnb-4bit"),
    )
    dataset_dir = DATA_DIR / dataset_subdir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_filename = os.environ.get("DPO_DATA_FILENAME", "arc_dpo_data_all.jsonl")
    dataset_path = dataset_dir / dataset_filename
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"missing prebuilt dataset: {dataset_path}\n"
            f"please run the corresponding build_dpo_rearc_vs_*.py script first"
        )

    base_output_dir = OUTPUT_DIR / output_subdir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Training with original rearc positives vs {negative_name} negatives")
    print(f"Dataset: {dataset_path}")
    print(f"Base model: {model_name_or_path}")
    print(f"Output: {base_output_dir}")
    print(f"Distributed: world_size={world_size}, rank={rank}, local_rank={local_rank}, device_map={device_map}")
    print("=" * 60 + "\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    tokenizer = load_local_tokenizer(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    resume_from_checkpoint_path = args.resume_from_checkpoint
    if resume_from_checkpoint_path and resume_from_checkpoint_path != "True" and os.path.isdir(resume_from_checkpoint_path):
        print(f"Loading LoRA from checkpoint: {resume_from_checkpoint_path}")
        model = PeftModel.from_pretrained(model, resume_from_checkpoint_path, is_trainable=True)
        print("Loaded LoRA from checkpoint")
    else:
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "embed_tokens", "lm_head",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print("Created new LoRA adapter")

    model.print_trainable_parameters()

    print(f"Loading dataset from: {dataset_path}")
    train_dataset = load_dpo_dataset(dataset_path)
    print(f"Dataset size: {len(train_dataset)} samples")

    per_device_train_batch_size = _get_int_env("DPO_PER_DEVICE_TRAIN_BATCH_SIZE", 2)
    gradient_accumulation_steps = _get_int_env("DPO_GRADIENT_ACCUMULATION_STEPS", 4)
    max_length = _get_int_env("DPO_MAX_LENGTH", 4096)
    max_prompt_length = _get_int_env("DPO_MAX_PROMPT_LENGTH", 3584)
    print(
        "DPO config overrides: "
        f"per_device_train_batch_size={per_device_train_batch_size}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}, "
        f"max_length={max_length}, "
        f"max_prompt_length={max_prompt_length}"
    )

    training_args = DPOConfig(
        output_dir=str(base_output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=1e-6,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=150,
        save_total_limit=None,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        beta=0.1,
        ddp_find_unused_parameters=False if world_size > 1 else None,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer_sig = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DPOTrainer(**trainer_kwargs)

    print("\nStarting training...")
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint == "True":
        resume_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    if rank == 0:
        model.save_pretrained(base_output_dir)
        tokenizer.save_pretrained(base_output_dir)
        print(f"\nTraining completed! Model saved to: {base_output_dir}")
    if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Llama-style ARC checkpoint with DPO.")
    parser.add_argument("--negative-name", default="output_transform", help="Label used in logs.")
    parser.add_argument("--task-limit", type=int, default=400, help="Kept for compatibility with experiment wrappers.")
    parser.add_argument("--dataset-subdir", required=True, help="Subdirectory under DIARC_DATA_DIR containing arc_dpo_data_all.jsonl.")
    parser.add_argument("--output-subdir", required=True, help="Subdirectory under DIARC_OUTPUT_DIR for the trained adapter.")
    args, remaining = parser.parse_known_args()

    import sys

    sys.argv = [sys.argv[0], *remaining]
    run_training(
        negative_name=args.negative_name,
        task_limit=args.task_limit,
        dataset_subdir=args.dataset_subdir,
        output_subdir=args.output_subdir,
    )


if __name__ == "__main__":
    main()

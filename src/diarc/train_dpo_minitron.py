import os
import sys
import json
import argparse
import inspect
import importlib.util
from pathlib import Path

# 设置离线模式，避免访问huggingface
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


def load_training_dependencies():
    try:
        import torch
        from datasets import Dataset
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer
        # peft 0.15.x 在某些 transformers 4.57.x 组合下会从顶层 transformers
        # 导入 BloomPreTrainedModel，但该符号不一定再暴露在顶层命名空间。
        # 对当前 Mistral 训练链来说，这个类不会被实际用到，补一个占位即可避免导入失败。
        if not hasattr(transformers, "BloomPreTrainedModel"):
            class BloomPreTrainedModel:  # pragma: no cover - compatibility shim
                pass
            transformers.BloomPreTrainedModel = BloomPreTrainedModel
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
        from trl import DPOTrainer, DPOConfig
    except (ModuleNotFoundError, ImportError) as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise ImportError(
            f"training dependency import failed: {missing}\n"
            f"please verify this environment has compatible versions of:\n"
            f"  torch datasets transformers peft trl bitsandbytes accelerate"
        ) from exc

    # 修复 trl 和 transformers 版本不兼容
    _original_dpo_log = DPOTrainer.log
    _original_dpo_get_batch_samples = DPOTrainer.get_batch_samples
    _original_dpo_compute_loss = DPOTrainer.compute_loss
    _original_dpo_create_model_card = DPOTrainer.create_model_card

    def _patched_dpo_log(self, logs, start_time=None):
        return _original_dpo_log(self, logs)

    def _patched_dpo_get_batch_samples(self, *args, **kwargs):
        if len(args) == 3:
            return Trainer.get_batch_samples(self, *args, **kwargs)
        return _original_dpo_get_batch_samples(self, *args, **kwargs)

    def _patched_dpo_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return _original_dpo_compute_loss(self, model, inputs, return_outputs=return_outputs)

    def _patched_dpo_create_model_card(self, *args, **kwargs):
        try:
            return _original_dpo_create_model_card(self, *args, **kwargs)
        except Exception as exc:
            print(f"Warning: failed to create DPO model card: {exc}")
            return None

    DPOTrainer.log = _patched_dpo_log
    DPOTrainer.get_batch_samples = _patched_dpo_get_batch_samples
    DPOTrainer.compute_loss = _patched_dpo_compute_loss
    DPOTrainer.create_model_card = _patched_dpo_create_model_card
    return {
        "torch": torch,
        "Dataset": Dataset,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Trainer": Trainer,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "PeftModel": PeftModel,
        "DPOTrainer": DPOTrainer,
        "DPOConfig": DPOConfig,
    }


def load_dpo_dataset(path: Path):
    deps = load_training_dependencies()
    Dataset = deps["Dataset"]
    data = []
    skipped_same = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["chosen"] == item["rejected"]:
                skipped_same += 1
                continue
            data.append(
                {
                    "prompt": item["instruction"] + item["input"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                }
            )
    if skipped_same:
        print(f"Filtered {skipped_same} chosen==rejected samples from {path}")
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
    AutoTokenizer = deps["AutoTokenizer"]
    BitsAndBytesConfig = deps["BitsAndBytesConfig"]
    LoraConfig = deps["LoraConfig"]
    get_peft_model = deps["get_peft_model"]
    prepare_model_for_kbit_training = deps["prepare_model_for_kbit_training"]
    PeftModel = deps["PeftModel"]
    DPOTrainer = deps["DPOTrainer"]
    DPOConfig = deps["DPOConfig"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="从checkpoint恢复训练，可指定路径或设置为'True'自动查找最新checkpoint")
    args, _ = parser.parse_known_args()

    model_name_or_path = os.environ.get(
        "BASE_MODEL_PATH",
        str(MODEL_DIR / "Mistral-NeMo-Minitron-8B-ARChitects-ReArc1200-bnb-4bit"),
    )
    dataset_dir = DATA_DIR / dataset_subdir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "arc_dpo_data_all.jsonl"
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
    print(f"Output: {base_output_dir}")
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
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                            "embed_tokens", "lm_head"],
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

    training_args = DPOConfig(
        output_dir=str(base_output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
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
        max_length=4096,
        max_prompt_length=3584,
        beta=0.1,
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

    model.save_pretrained(base_output_dir)
    tokenizer.save_pretrained(base_output_dir)
    print(f"\nTraining completed! Model saved to: {base_output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a NeMo-Minitron ARC checkpoint with DPO.")
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

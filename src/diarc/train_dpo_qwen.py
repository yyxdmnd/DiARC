import argparse
import inspect
from pathlib import Path

try:
    from .paths import DATA_DIR, OUTPUT_DIR
    from .train_dpo_qwen_common import (
        DEFAULT_BASE_MODEL,
        _build_quant_config,
        _supports_bf16,
        load_dpo_dataset,
        load_local_tokenizer,
        load_training_dependencies,
    )
except ImportError:  # pragma: no cover
    from paths import DATA_DIR, OUTPUT_DIR
    from train_dpo_qwen_common import (
        DEFAULT_BASE_MODEL,
        _build_quant_config,
        _supports_bf16,
        load_dpo_dataset,
        load_local_tokenizer,
        load_training_dependencies,
    )

"""DPO training entry point for the Qwen3-4B ARC-specialized checkpoint."""


def run_training(
    negative_name: str,
    dataset_subdir: str,
    output_subdir: str,
    base_model_path: str | None = None,
    load_in_4bit: bool = False,
    resume_from_checkpoint: str | None = None,
    lora_r: int = 256,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> None:
    deps = load_training_dependencies()
    torch = deps["torch"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    LoraConfig = deps["LoraConfig"]
    get_peft_model = deps["get_peft_model"]
    prepare_model_for_kbit_training = deps["prepare_model_for_kbit_training"]
    PeftModel = deps["PeftModel"]
    DPOTrainer = deps["DPOTrainer"]
    DPOConfig = deps["DPOConfig"]

    model_name_or_path = str(Path(base_model_path or DEFAULT_BASE_MODEL).resolve())
    dataset_dir = DATA_DIR / dataset_subdir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "arc_dpo_data_all.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"missing prebuilt dataset: {dataset_path}\n"
            "please build or place arc_dpo_data_all.jsonl first"
        )

    base_output_dir = OUTPUT_DIR / output_subdir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Training Qwen DPO model on {negative_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Base model: {model_name_or_path}")
    print(f"Output: {base_output_dir}")
    print(f"4-bit load: {load_in_4bit}")
    print(
        f"LoRA params: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}"
    )
    print("=" * 60 + "\n")

    quant_args = argparse.Namespace(load_in_4bit=load_in_4bit)
    quantization_config = _build_quant_config(quant_args, deps, torch)
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
        "attn_implementation": "sdpa",
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = False

    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    tokenizer = load_local_tokenizer(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    resume_from_checkpoint_path = resume_from_checkpoint
    if (
        resume_from_checkpoint_path
        and resume_from_checkpoint_path != "True"
        and Path(resume_from_checkpoint_path).is_dir()
    ):
        print(f"Loading LoRA from checkpoint: {resume_from_checkpoint_path}")
        model = PeftModel.from_pretrained(
            model,
            resume_from_checkpoint_path,
            is_trainable=True,
        )
        print("Loaded LoRA from checkpoint")
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "embed_tokens",
                "lm_head",
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print("Created new LoRA adapter")

    model.print_trainable_parameters()

    train_dataset = load_dpo_dataset(dataset_path)
    print(f"Dataset size: {len(train_dataset)} samples")

    bf16 = _supports_bf16(torch)
    training_args = DPOConfig(
        output_dir=str(base_output_dir),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=150,
        save_total_limit=None,
        bf16=bf16,
        fp16=not bf16,
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

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    trainer_sig = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DPOTrainer(**trainer_kwargs)

    print("\nStarting training...")
    resume_checkpoint = resume_from_checkpoint
    if resume_checkpoint == "True":
        resume_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    model.save_pretrained(base_output_dir)
    tokenizer.save_pretrained(base_output_dir)
    print(f"\nTraining completed! Model saved to: {base_output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Qwen ARC-specialized checkpoint with DPO.")
    parser.add_argument("--negative-name", default="output_transform", help="Label used in logs.")
    parser.add_argument("--dataset-subdir", required=True, help="Subdirectory under DIARC_DATA_DIR containing arc_dpo_data_all.jsonl.")
    parser.add_argument("--output-subdir", required=True, help="Subdirectory under DIARC_OUTPUT_DIR for the trained adapter.")
    parser.add_argument(
        "--base-model-path",
        default=str(DEFAULT_BASE_MODEL),
        help="Local Qwen base/SFT checkpoint path.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the Qwen checkpoint with NF4 4-bit quantization before DPO training.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Resume from a checkpoint path, or set to 'True' for trainer auto-resume.",
    )
    parser.add_argument("--lora-r", type=int, default=256)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()
    run_training(
        negative_name=args.negative_name,
        dataset_subdir=args.dataset_subdir,
        output_subdir=args.output_subdir,
        base_model_path=args.base_model_path,
        load_in_4bit=args.load_in_4bit,
        resume_from_checkpoint=args.resume_from_checkpoint,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


if __name__ == "__main__":
    main()

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path

# Force fully offline behavior for local training.
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

DEFAULT_BASE_MODEL = (
    MODEL_DIR / "qwen3_4b_grids15_sft139_bfloat16"
)


def load_local_tokenizer(model_path: str):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    model_dir = Path(model_path)
    tokenizer_file = model_dir / "tokenizer.json"
    tokenizer_config_file = model_dir / "tokenizer_config.json"
    special_tokens_file = model_dir / "special_tokens_map.json"
    added_tokens_file = model_dir / "added_tokens.json"
    vocab_file = model_dir / "vocab.json"
    config_file = model_dir / "config.json"

    def _load_json(path: Path):
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def _normalize_special_token(value):
        if isinstance(value, dict) and "content" in value:
            value = value["content"]
        if isinstance(value, str) and value:
            return value
        return None

    tokenizer_config = _load_json(tokenizer_config_file)
    special_tokens = _load_json(special_tokens_file)
    tokenizer_json = _load_json(tokenizer_file)
    added_tokens = _load_json(added_tokens_file)
    vocab = _load_json(vocab_file)
    model_config = _load_json(config_file)

    def _build_tokenizer_kwargs(unk_token_override=None):
        kwargs = {}
        for key in [
            "bos_token",
            "eos_token",
            "unk_token",
            "pad_token",
            "cls_token",
            "sep_token",
            "mask_token",
        ]:
            value = special_tokens.get(key, tokenizer_config.get(key))
            if key == "unk_token":
                value = unk_token_override or value or tokenizer_json.get("model", {}).get("unk_token")
            value = _normalize_special_token(value)
            if value is not None:
                kwargs[key] = value

        for key in [
            "model_max_length",
            "padding_side",
            "truncation_side",
            "clean_up_tokenization_spaces",
        ]:
            value = tokenizer_config.get(key)
            if value is not None:
                kwargs[key] = value
        return kwargs

    def _wordlevel_needs_backend_patch():
        model_data = tokenizer_json.get("model", {})
        if model_data.get("type") != "WordLevel":
            return False
        unk_token = _normalize_special_token(
            special_tokens.get("unk_token")
            or tokenizer_config.get("unk_token")
            or model_data.get("unk_token")
        )
        if unk_token is None:
            return False
        return unk_token not in model_data.get("vocab", {})

    def _build_patched_wordlevel_tokenizer():
        from tokenizers import Tokenizer

        patched_json = json.loads(json.dumps(tokenizer_json))
        model_data = patched_json.setdefault("model", {})
        model_vocab = model_data.setdefault("vocab", {})
        unk_token = _normalize_special_token(
            special_tokens.get("unk_token")
            or tokenizer_config.get("unk_token")
            or model_data.get("unk_token")
            or special_tokens.get("pad_token")
            or tokenizer_config.get("pad_token")
            or special_tokens.get("eos_token")
            or tokenizer_config.get("eos_token")
        )
        if unk_token is None:
            unk_token = "<|endoftext|>"

        unk_token_id = added_tokens.get(unk_token)
        if unk_token_id is None:
            unk_token_id = vocab.get(unk_token)
        if unk_token_id is None:
            for token_data in patched_json.get("added_tokens", []):
                if token_data.get("content") == unk_token and token_data.get("id") is not None:
                    unk_token_id = int(token_data["id"])
                    break
        if unk_token_id is None:
            unk_token_id = max(model_vocab.values(), default=-1) + 1

        model_vocab.setdefault(unk_token, int(unk_token_id))
        model_data["unk_token"] = unk_token

        backend = Tokenizer.from_str(json.dumps(patched_json))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            **_build_tokenizer_kwargs(unk_token_override=unk_token),
        )
        chat_template = tokenizer_config.get("chat_template")
        if chat_template:
            tokenizer.chat_template = chat_template
        return tokenizer

    def _repair_special_tokens(tokenizer):
        id_to_token = {int(v): k for k, v in added_tokens.items()}
        id_to_token.update({int(v): k for k, v in vocab.items()})
        for token_data in tokenizer_json.get("added_tokens", []):
            token_id = token_data.get("id")
            token = token_data.get("content")
            if token_id is not None and token:
                id_to_token[int(token_id)] = token

        def _token_from_id(token_id):
            if token_id is None:
                return None
            if tokenizer.pad_token_id == token_id and tokenizer.pad_token is not None:
                return tokenizer.pad_token
            if tokenizer.eos_token_id == token_id and tokenizer.eos_token is not None:
                return tokenizer.eos_token
            return id_to_token.get(int(token_id))

        def _set_missing(attr_name, *candidates):
            if getattr(tokenizer, attr_name) is not None:
                return
            for candidate in candidates:
                token = _normalize_special_token(candidate)
                if token is not None:
                    setattr(tokenizer, attr_name, token)
                    return

        _set_missing(
            "bos_token",
            special_tokens.get("bos_token"),
            tokenizer_config.get("bos_token"),
            _token_from_id(model_config.get("bos_token_id")),
        )
        _set_missing(
            "eos_token",
            special_tokens.get("eos_token"),
            tokenizer_config.get("eos_token"),
            _token_from_id(model_config.get("eos_token_id")),
        )
        _set_missing(
            "pad_token",
            special_tokens.get("pad_token"),
            tokenizer_config.get("pad_token"),
            _token_from_id(model_config.get("pad_token_id")),
            tokenizer.eos_token,
        )
        _set_missing(
            "unk_token",
            special_tokens.get("unk_token"),
            tokenizer_config.get("unk_token"),
            tokenizer_json.get("model", {}).get("unk_token"),
            _token_from_id(model_config.get("unk_token_id")),
            tokenizer.pad_token,
            tokenizer.eos_token,
        )
        return tokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer = _repair_special_tokens(tokenizer)
        if _wordlevel_needs_backend_patch():
            return _repair_special_tokens(_build_patched_wordlevel_tokenizer())
        return tokenizer
    except AttributeError as exc:
        if "'dict' object has no attribute 'model_type'" not in str(exc):
            raise

    if not tokenizer_file.exists():
        raise FileNotFoundError(f"missing tokenizer file: {tokenizer_file}")

    if _wordlevel_needs_backend_patch():
        tokenizer = _build_patched_wordlevel_tokenizer()
    else:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            **_build_tokenizer_kwargs(),
        )
    chat_template = tokenizer_config.get("chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
    return _repair_special_tokens(tokenizer)


def load_training_dependencies():
    try:
        import torch
        from datasets import Dataset
        import transformers
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
        )

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
            "please verify this environment has compatible versions of:\n"
            "  torch datasets transformers peft trl bitsandbytes accelerate"
        ) from exc

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
    skipped_same = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
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


def get_available_stages(dataset_dir: Path) -> list[int]:
    stages = []
    for stage in range(1, 33):
        stage_path = dataset_dir / f"arc_dpo_data_stage_{stage}.jsonl"
        if not stage_path.exists():
            continue
        if stage_path.stat().st_size == 0:
            continue
        stages.append(stage)
    return stages


def _supports_bf16(torch) -> bool:
    if not torch.cuda.is_available():
        return False
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            return True
    return True


def _build_quant_config(args, deps, torch):
    if not args.load_in_4bit:
        return None

    BitsAndBytesConfig = deps["BitsAndBytesConfig"]
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def train_stage(
    stage: int,
    first_stage: int,
    dataset_dir: Path,
    base_output_dir: Path,
    args,
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

    model_name_or_path = str(Path(args.base_model_path).resolve())
    dataset_path = dataset_dir / f"arc_dpo_data_stage_{stage}.jsonl"
    output_dir = base_output_dir / f"stage_{stage}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prev_stage_dir = base_output_dir / f"stage_{stage - 1}"

    print("\n" + "=" * 60)
    print(f"Stage {stage}: {dataset_path.name}")
    print(f"Dataset: {dataset_path}")
    print(f"Base model: {model_name_or_path}")
    print(f"Output: {output_dir}")
    print("=" * 60 + "\n")

    quantization_config = _build_quant_config(args, deps, torch)
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

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    tokenizer = load_local_tokenizer(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    resume_from_checkpoint_path = args.resume_from_checkpoint
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
        print(f"Stage {stage}: loaded LoRA from checkpoint")
    elif stage == first_stage:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
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
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print(f"Stage {stage}: created new LoRA adapter")
    else:
        if not prev_stage_dir.is_dir():
            raise FileNotFoundError(f"missing previous stage adapter: {prev_stage_dir}")
        print(f"Loading LoRA from previous stage: {prev_stage_dir}")
        model = PeftModel.from_pretrained(model, prev_stage_dir, is_trainable=True)
        print(f"Stage {stage}: loaded LoRA from stage {stage - 1}")

    model.print_trainable_parameters()

    train_dataset = load_dpo_dataset(dataset_path)
    print(f"Dataset size: {len(train_dataset)} samples")

    bf16 = _supports_bf16(torch)
    training_args = DPOConfig(
        output_dir=str(output_dir),
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

    print(f"\nStarting Stage {stage} training...")
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint == "True":
        resume_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nStage {stage} completed! Model saved to: {output_dir}")


def run_staged_training(
    negative_name: str,
    dataset_subdir: str,
    output_subdir: str,
    lora_r: int = 256,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Only train one stage; default is to train every non-empty stage in order.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint path, or set to 'True' to let trainer resume automatically.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=str(DEFAULT_BASE_MODEL),
        help="Local Qwen model path. Default is the local qwen3_4b_grids15_sft139_bfloat16 checkpoint.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the Qwen checkpoint with NF4 4-bit quantization before DPO training.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=lora_r,
        help="LoRA rank used when creating a new stage-1 adapter.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=lora_alpha,
        help="LoRA alpha used when creating a new stage-1 adapter.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=lora_dropout,
        help="LoRA dropout used when creating a new stage-1 adapter.",
    )
    args = parser.parse_args()

    dataset_dir = DATA_DIR / dataset_subdir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"missing dataset dir: {dataset_dir}")

    available_stages = get_available_stages(dataset_dir)
    if not available_stages:
        raise FileNotFoundError(f"no non-empty stage files found in: {dataset_dir}")

    base_output_dir = OUTPUT_DIR / output_subdir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Training Qwen DPO model: {negative_name}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Base model: {Path(args.base_model_path).resolve()}")
    print(f"Output dir: {base_output_dir}")
    print(f"Available stages: {available_stages}")
    print(f"4-bit load: {args.load_in_4bit}")
    print(
        f"LoRA params: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}"
    )
    print("=" * 60 + "\n")

    if args.stage is not None:
        if args.stage not in available_stages:
            raise ValueError(f"stage {args.stage} is not available in {dataset_dir}")
        train_stage(
            stage=args.stage,
            first_stage=min(available_stages),
            dataset_dir=dataset_dir,
            base_output_dir=base_output_dir,
            args=args,
        )
        return

    first_stage = min(available_stages)
    for stage in available_stages:
        train_stage(
            stage=stage,
            first_stage=first_stage,
            dataset_dir=dataset_dir,
            base_output_dir=base_output_dir,
            args=args,
        )

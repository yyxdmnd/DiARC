#!/usr/bin/env python3
"""Shared helpers for protocol-fixed Llama direct ARC-AGI1 evaluations."""

from __future__ import annotations

import bz2
import gc
import importlib.util
import json
import os
import pickle
from pathlib import Path
from typing import Optional

from datasets import Dataset
from diskcache import Cache
from tqdm import tqdm


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", "/tmp/unsloth_compiled_cache")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

_raw_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
if _raw_visible_devices:
    _first_visible_device = _raw_visible_devices.split(",")[0].strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = _first_visible_device

for _dist_env_name in [
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "ACCELERATE_PROCESS_INDEX",
    "ACCELERATE_LOCAL_PROCESS_INDEX",
    "ACCELERATE_NUM_PROCESSES",
    "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_USE_FSDP",
]:
    os.environ.pop(_dist_env_name, None)

_real_find_spec = importlib.util.find_spec


def _patched_find_spec(name, package=None):
    if name == "torchao" or name.startswith("torchao."):
        return None
    return _real_find_spec(name, package)


importlib.util.find_spec = _patched_find_spec

import torch  # noqa: E402
import torch._inductor as _torch_inductor  # noqa: E402

if torch.cuda.is_available():
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass

try:
    import torch._inductor.config as _torch_inductor_config  # noqa: E402

    if not hasattr(_torch_inductor, "config"):
        _torch_inductor.config = _torch_inductor_config
except Exception:
    pass

from unsloth import FastLanguageModel  # noqa: E402
from unsloth import UnslothTrainer as Trainer  # noqa: E402
from unsloth import UnslothTrainingArguments as TrainingArguments  # noqa: E402
from unsloth import is_bfloat16_supported, unsloth_train  # noqa: E402

try:
    from .arc_downloader import download_arc_data  # noqa: E402
    from .arc_loader import ArcDataset  # noqa: E402
    from .inference_tools import inference_run  # noqa: E402
    from .model_tools import InputMaskingDataCollator, load_peft_state, load_unsloth_model  # noqa: E402
    from .paths import DATA_DIR, MODEL_DIR, OUTPUT_DIR  # noqa: E402
    from .selection import EvalTool, max_aug_prob, max_gen_prob, min_aug_prob, sum_aug_prob  # noqa: E402
except ImportError:  # pragma: no cover
    from arc_downloader import download_arc_data  # noqa: E402
    from arc_loader import ArcDataset  # noqa: E402
    from inference_tools import inference_run  # noqa: E402
    from model_tools import InputMaskingDataCollator, load_peft_state, load_unsloth_model  # noqa: E402
    from paths import DATA_DIR, MODEL_DIR, OUTPUT_DIR  # noqa: E402
    from selection import EvalTool, max_aug_prob, max_gen_prob, min_aug_prob, sum_aug_prob  # noqa: E402


DEFAULT_BASE_MODEL = MODEL_DIR / "Llama-3.2-3B-ReArc-merged"
DEFAULT_ARC_DATA = DATA_DIR / "ARC-AGI-1"
LORA_LAYERS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "embed_tokens",
    "lm_head",
]


def get_env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name)
    return Path(value) if value is not None and value != "" else Path(default)


def get_env_optional_path(name: str, default: Optional[Path | str]) -> Optional[Path]:
    value = os.environ.get(name)
    if value is None:
        return Path(default) if default is not None else None
    if value.strip().lower() in {"", "none", "null", "no", "false", "0"}:
        return None
    return Path(value)


def get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_single_device_map():
    if not torch.cuda.is_available():
        return None
    return {"": 0}


def disable_gradient_checkpointing(module):
    if hasattr(module, "gradient_checkpointing_disable"):
        try:
            module.gradient_checkpointing_disable()
        except Exception:
            pass
    if hasattr(module, "config"):
        module.config.use_cache = True
    for submodule in module.modules():
        if hasattr(submodule, "gradient_checkpointing"):
            submodule.gradient_checkpointing = False


def cleanup_model(model_tok):
    if model_tok is None:
        return
    model, tokenizer = model_tok
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def build_fmt_opts(tokenizer):
    return dict(
        preprompt="ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz",
        query_beg="I",
        reply_beg="\n+/-=O",
        reply_end="\n" + tokenizer.eos_token,
        lines_sep="\n",
        max_tokens=16384,
    )


def load_eval_dataset(arc_data_path: Path, eval_task_limit: int) -> ArcDataset:
    challenge_file = os.environ.get("ARC_CHALLENGES_FILE", "arc-agi_evaluation_challenges.json")
    solution_file = os.environ.get("ARC_SOLUTIONS_FILE", "arc-agi_evaluation_solutions.json")
    challenge_path = arc_data_path / challenge_file
    solution_path = arc_data_path / solution_file
    if not (challenge_path.is_file() and solution_path.is_file()):
        download_arc_data(str(arc_data_path))
    eval_dataset = ArcDataset.load_from_json(str(challenge_path))
    eval_dataset = eval_dataset.load_solutions(str(solution_path))
    if eval_task_limit > 0:
        selected_base_keys = set(sorted(eval_dataset.challenge.keys())[:eval_task_limit])
        selected_keys = [key for key in eval_dataset.keys if eval_dataset.get_base_key(key) in selected_base_keys]
        eval_dataset = eval_dataset.change_keys(selected_keys)
    return eval_dataset


def make_eval_tool(use_aug_score: bool, n_guesses: int) -> EvalTool:
    if use_aug_score:
        return EvalTool(
            n_guesses=n_guesses,
            score_algos=[max_gen_prob, max_aug_prob, min_aug_prob, sum_aug_prob],
            sorting_algo="sum_aug_prob",
        )
    return EvalTool(
        n_guesses=n_guesses,
        score_algos=[max_gen_prob],
        sorting_algo="max_gen_prob",
    )


def write_outputs(output_path: Path, eval_dataset: ArcDataset, inference_keys: dict, inference_results: dict):
    results_file = output_path / "results.pickle.bz2"
    with bz2.BZ2File(results_file, "w") as handle:
        pickle.dump(inference_keys, handle)
        pickle.dump(inference_results, handle)

    submission_file = output_path / "submission.json"
    with submission_file.open("w", encoding="utf-8") as handle:
        json.dump(eval_dataset.get_submission(inference_results), handle)
    with submission_file.open("r", encoding="utf-8") as handle:
        score = eval_dataset.validate_submission(json.load(handle))
    print(f"Reload score for '{submission_file}': {score}")


def load_lora_adapter_if_needed(model, lora_adapter_path: Optional[Path]):
    if lora_adapter_path is None:
        return model
    if not lora_adapter_path.is_dir():
        raise FileNotFoundError(f"LORA_ADAPTER_PATH does not exist: {lora_adapter_path}")
    model = FastLanguageModel.get_peft_model(
        model=model,
        target_modules=LORA_LAYERS,
        r=32,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    load_peft_state(model, str(lora_adapter_path))
    print(f"Loaded LoRA adapter from: {lora_adapter_path}")
    return model


def sync_generation_special_tokens(model, tokenizer):
    configs = [
        getattr(model, "config", None),
        getattr(model, "generation_config", None),
    ]
    for name in ["eos_token_id", "pad_token_id", "bos_token_id"]:
        fallback_token_id = getattr(tokenizer, name, None)
        for config in configs:
            if config is None:
                continue
            token_id = getattr(config, name, None)
            if isinstance(token_id, (list, tuple, set)):
                token_id = [item for item in token_id if item is not None]
                token_id = token_id if token_id else None
            if token_id is None and fallback_token_id is not None:
                token_id = fallback_token_id
            if token_id is not None:
                setattr(config, name, token_id)


def run_direct_eval(
    *,
    protocol_name: str,
    output_subdir: str,
    use_dfs: bool,
    use_aug_score: bool,
    input_aug_n: int,
    num_return_sequences: int = 2,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.9,
    min_prob: float = 0.09,
    pass_guess: bool = True,
):
    base_model = get_env_path("BASE_MODEL_PATH", DEFAULT_BASE_MODEL)
    lora_adapter_path = get_env_optional_path("LORA_ADAPTER_PATH", None)
    arc_data_path = get_env_path("ARC_DATA_PATH", DEFAULT_ARC_DATA)
    output_path = get_env_path("EVAL_OUTPUT_PATH", OUTPUT_DIR / output_subdir)
    bits = get_env_int("LOAD_BITS", 4)
    eval_task_limit = get_env_int("EVAL_TASK_LIMIT", 0)
    use_cache = get_env_bool("DIRECT_USE_CACHE", False)

    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(protocol_name)
    print(f"Base model:      {base_model}")
    print(f"LoRA adapter:    {lora_adapter_path if lora_adapter_path is not None else '(none)'}")
    print(f"ARC data:        {arc_data_path}")
    print(f"Output:          {output_path}")
    print(f"bits:            {bits}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"device_map:      {get_single_device_map()}")
    decode_mode = "DFS" if use_dfs else ("sample generate" if do_sample else "greedy generate")
    print(f"decode:          {decode_mode}")
    print(f"min_prob:        {min_prob if use_dfs else '(disabled)'}")
    print(f"num_return_seq:  {num_return_sequences if not use_dfs else '(disabled)'}")
    print(f"do_sample:       {do_sample if not use_dfs else '(disabled)'}")
    print(f"temperature:     {temperature if (not use_dfs and do_sample) else '(disabled)'}")
    print(f"top_p:           {top_p if (not use_dfs and do_sample) else '(disabled)'}")
    print(f"aug scoring:     {'enabled' if use_aug_score else 'disabled'}")
    print(f"input aug n:     {input_aug_n}")
    print("Disabled:        TTT")
    print("=" * 80)

    eval_dataset = load_eval_dataset(arc_data_path, eval_task_limit)
    model, tokenizer = load_unsloth_model(str(base_model), bits=bits, device_map=get_single_device_map())
    model = load_lora_adapter_if_needed(model, lora_adapter_path)
    sync_generation_special_tokens(model, tokenizer)

    FastLanguageModel.for_inference(model)
    disable_gradient_checkpointing(model)
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True

    fmt_opts = build_fmt_opts(tokenizer)
    eval_tool = make_eval_tool(use_aug_score=use_aug_score, n_guesses=2)
    inference_results = {}
    inference_keys = {}

    cache_decorator = None
    if use_cache:
        cache_decorator = Cache(str(output_path / "direct.cache")).memoize(
            typed=True,
            ignore=set(["model_tok", "guess"]),
        )

    with tqdm(total=len(eval_dataset.grouped_keys()), desc="direct inference") as pbar:
        for global_i, (base_key, task_groups) in enumerate(eval_dataset.grouped_keys().items()):
            part_dataset = eval_dataset.change_keys([key for group in task_groups for key in group])
            infer_aug_opts = dict(tp="all", rt="all", perm=True, shfl_ex=True, seed=10000 + global_i)
            inference_dataset = part_dataset.augment(n=input_aug_n, **infer_aug_opts) if input_aug_n > 1 else part_dataset

            for grouped_base_key, grouped_task_groups in inference_dataset.grouped_keys().items():
                inference_keys[grouped_base_key] = [key for group in grouped_task_groups for key in group]

            aug_score_opts = None
            if use_aug_score:
                aug_score_opts = dict(**infer_aug_opts, n=2)

            inference_kwargs = {}
            if use_dfs:
                inference_kwargs["min_prob"] = min_prob
            else:
                inference_kwargs["num_return_sequences"] = num_return_sequences
                inference_kwargs["do_sample"] = do_sample
                if do_sample:
                    inference_kwargs["temperature"] = temperature
                    inference_kwargs["top_p"] = top_p

            part_results = inference_run(
                model_tok=(model, tokenizer),
                fmt_opts=fmt_opts,
                dataset=inference_dataset,
                aug_score_opts=aug_score_opts,
                pass_guess=pass_guess,
                callback=eval_tool.process_result,
                cache=cache_decorator,
                print_func=pbar.write,
                **inference_kwargs,
            )
            inference_results.update(part_results)
            pbar.update(1)

    write_outputs(output_path, eval_dataset, inference_keys, inference_results)
    cleanup_model((model, tokenizer))


def load_task_state(path: Path):
    with bz2.BZ2File(path, "rb") as handle:
        return pickle.load(handle)


def save_task_state(path: Path, inference_keys_part, part_results):
    payload = {
        "inference_keys": inference_keys_part,
        "inference_results": part_results,
    }
    with bz2.BZ2File(path, "wb") as handle:
        pickle.dump(payload, handle)


def run_direct_ttt_only_eval(
    *,
    protocol_name: str,
    output_subdir: str,
    num_return_sequences: int = 2,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.9,
):
    base_model = get_env_path("BASE_MODEL_PATH", DEFAULT_BASE_MODEL)
    lora_adapter_path = get_env_optional_path("LORA_ADAPTER_PATH", None)
    arc_data_path = get_env_path("ARC_DATA_PATH", DEFAULT_ARC_DATA)
    output_path = get_env_path("EVAL_OUTPUT_PATH", OUTPUT_DIR / output_subdir)
    bits = get_env_int("LOAD_BITS", 4)
    eval_task_limit = get_env_int("EVAL_TASK_LIMIT", 0)
    use_cache = get_env_bool("DIRECT_USE_CACHE", False)

    output_path.mkdir(parents=True, exist_ok=True)
    task_state_dir = output_path / "task_states"
    task_state_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(protocol_name)
    print(f"Base model:      {base_model}")
    print(f"LoRA adapter:    {lora_adapter_path if lora_adapter_path is not None else '(fresh TTT adapter)'}")
    print(f"ARC data:        {arc_data_path}")
    print(f"Output:          {output_path}")
    print(f"bits:            {bits}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"device_map:      {get_single_device_map()}")
    print("decode:          sample generate")
    print(f"num_return_seq:  {num_return_sequences}")
    print(f"do_sample:       {do_sample}")
    print(f"temperature:     {temperature if do_sample else '(disabled)'}")
    print(f"top_p:           {top_p if do_sample else '(disabled)'}")
    print("TTT train aug n: 8")
    print("TTT lr:          1e-4")
    print("TTT emb lr:      1e-5")
    print("TTT epochs:      1")
    print("Disabled:        DFS, aug scoring, input augmentation")
    print("=" * 80)

    eval_dataset = load_eval_dataset(arc_data_path, eval_task_limit)
    tokenizer_model, tokenizer = load_unsloth_model(str(base_model), bits=bits, device_map=get_single_device_map())
    sync_generation_special_tokens(tokenizer_model, tokenizer)
    fmt_opts = build_fmt_opts(tokenizer)
    del tokenizer_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_tool = make_eval_tool(use_aug_score=False, n_guesses=2)
    inference_results = {}
    inference_keys = {}
    completed_base_keys = set()

    results_file = output_path / "results.pickle.bz2"
    if results_file.is_file():
        try:
            with bz2.BZ2File(results_file, "rb") as handle:
                saved_inference_keys = pickle.load(handle)
                saved_inference_results = pickle.load(handle)
            if isinstance(saved_inference_keys, dict):
                inference_keys.update(saved_inference_keys)
            if isinstance(saved_inference_results, dict):
                inference_results.update(saved_inference_results)
                completed_base_keys.update(saved_inference_results.keys())
            print(f"Loaded existing aggregate results from: {results_file}")
        except Exception as exc:
            print(f"Could not load existing aggregate results from {results_file}: {exc}")

    for cached_state_path in sorted(task_state_dir.glob("*.pkl.bz2")):
        try:
            payload = load_task_state(cached_state_path)
            if not isinstance(payload, dict):
                continue
            saved_inference_keys = payload.get("inference_keys", {})
            saved_inference_results = payload.get("inference_results", {})
            if isinstance(saved_inference_keys, dict):
                inference_keys.update(saved_inference_keys)
            if isinstance(saved_inference_results, dict):
                inference_results.update(saved_inference_results)
                completed_base_keys.update(saved_inference_results.keys())
        except Exception as exc:
            print(f"Skipping unreadable task cache '{cached_state_path}': {exc}")

    split_parts = list(eval_dataset.split(n=len(eval_dataset.challenge), split_seed=123))

    with tqdm(list(enumerate(split_parts)), desc="direct+ttt-only inference") as pbar:
        for global_i, eval_dataset_part in pbar:
            base_key = sorted(eval_dataset_part.challenge.keys())[0]
            if base_key in completed_base_keys:
                pbar.write(f"[Direct+TTT-only] skipping completed task: {base_key}")
                continue

            model_tok_cache = [None]

            def get_model_and_tokenizer(cache=model_tok_cache):
                if cache[0] is not None:
                    return cache[0]

                model, task_tokenizer = load_unsloth_model(
                    str(base_model),
                    bits=bits,
                    device_map=get_single_device_map(),
                )
                sync_generation_special_tokens(model, task_tokenizer)
                model = FastLanguageModel.get_peft_model(
                    model=model,
                    target_modules=LORA_LAYERS,
                    r=32,
                    lora_alpha=16,
                    lora_dropout=0,
                    bias="none",
                    use_gradient_checkpointing=True,
                    random_state=42,
                    use_rslora=True,
                    loftq_config=None,
                )

                if lora_adapter_path is not None:
                    if not lora_adapter_path.is_dir():
                        raise FileNotFoundError(f"LORA_ADAPTER_PATH does not exist: {lora_adapter_path}")
                    load_peft_state(model, str(lora_adapter_path))
                    pbar.write(f"[Direct+TTT-only] loaded initial adapter: {lora_adapter_path}")

                train_aug_opts = dict(tp="all", rt="all", shfl_keys=True, perm=True, shfl_ex=True, seed=global_i)
                train_dataset_aug = eval_dataset_part.remove_test_data().augment(n=8, **train_aug_opts)
                train_dataset_as_list = train_dataset_aug.as_list(len_name="text", **fmt_opts)

                FastLanguageModel.for_training(model)
                trainer = Trainer(
                    model=model,
                    tokenizer=task_tokenizer,
                    train_dataset=Dataset.from_list(train_dataset_as_list),
                    dataset_text_field="text",
                    max_seq_length=fmt_opts["max_tokens"],
                    data_collator=InputMaskingDataCollator(
                        instruction_template=fmt_opts["query_beg"],
                        response_template=fmt_opts["reply_beg"],
                        mlm=False,
                        tokenizer=task_tokenizer,
                        mask_first_n_examples=1,
                    ),
                    args=TrainingArguments(
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=1,
                        warmup_steps=32,
                        num_train_epochs=1,
                        learning_rate=1e-4,
                        embedding_learning_rate=1e-5,
                        fp16=not is_bfloat16_supported(),
                        bf16=is_bfloat16_supported(),
                        logging_steps=8,
                        optim="adamw_8bit",
                        weight_decay=0.00,
                        lr_scheduler_type="cosine",
                        seed=42,
                        output_dir="tmp_output",
                        save_strategy="no",
                        report_to="none",
                    ),
                )
                unsloth_train(trainer)
                FastLanguageModel.for_inference(model)
                disable_gradient_checkpointing(model)
                if hasattr(model, "generation_config"):
                    model.generation_config.use_cache = True
                cache[0] = (model, task_tokenizer)
                return cache[0]

            inference_dataset = eval_dataset_part
            for grouped_base_key, task_groups in inference_dataset.grouped_keys().items():
                inference_keys[grouped_base_key] = [key for group in task_groups for key in group]

            cache_decorator = None
            if use_cache:
                cache_decorator = Cache(str(output_path / f"{base_key}.cache")).memoize(
                    typed=True,
                    ignore=set(["model_tok", "guess"]),
                )

            part_results = inference_run(
                model_tok=get_model_and_tokenizer,
                fmt_opts=fmt_opts,
                dataset=inference_dataset,
                pass_guess=False,
                callback=eval_tool.process_result,
                cache=cache_decorator,
                print_func=pbar.write,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
            )
            inference_results.update(part_results)
            save_task_state(task_state_dir / f"{base_key}.pkl.bz2", {base_key: inference_keys[base_key]}, part_results)
            completed_base_keys.add(base_key)

            cleanup_model(model_tok_cache[0])
            model_tok_cache[0] = None

    write_outputs(output_path, eval_dataset, inference_keys, inference_results)


def run_direct_ttt_eval(
    *,
    protocol_name: str,
    output_subdir: str,
    use_dfs: bool,
    use_aug_score: bool,
    input_aug_n: int,
    num_return_sequences: int = 2,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.9,
    min_prob: float = 0.09,
    pass_guess: bool = False,
    ttt_aug_n: int = 0,
    ttt_learning_rate: float = 1e-4,
    ttt_embedding_learning_rate: float = 1e-5,
    ttt_num_epochs: float = 1.0,
    ttt_warmup_steps: int = 32,
):
    base_model = get_env_path("BASE_MODEL_PATH", DEFAULT_BASE_MODEL)
    lora_adapter_path = get_env_optional_path("LORA_ADAPTER_PATH", None)
    arc_data_path = get_env_path("ARC_DATA_PATH", DEFAULT_ARC_DATA)
    output_path = get_env_path("EVAL_OUTPUT_PATH", OUTPUT_DIR / output_subdir)
    bits = get_env_int("LOAD_BITS", 4)
    eval_task_limit = get_env_int("EVAL_TASK_LIMIT", 0)
    use_cache = get_env_bool("DIRECT_USE_CACHE", False)

    output_path.mkdir(parents=True, exist_ok=True)
    task_state_dir = output_path / "task_states"
    task_state_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(protocol_name)
    print(f"Base model:      {base_model}")
    print(f"LoRA adapter:    {lora_adapter_path if lora_adapter_path is not None else '(fresh TTT adapter)'}")
    print(f"ARC data:        {arc_data_path}")
    print(f"Output:          {output_path}")
    print(f"bits:            {bits}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"device_map:      {get_single_device_map()}")
    decode_mode = "DFS" if use_dfs else ("sample generate" if do_sample else "greedy generate")
    print(f"decode:          {decode_mode}")
    print(f"min_prob:        {min_prob if use_dfs else '(disabled)'}")
    print(f"num_return_seq:  {num_return_sequences if not use_dfs else '(disabled)'}")
    print(f"do_sample:       {do_sample if not use_dfs else '(disabled)'}")
    print(f"temperature:     {temperature if (not use_dfs and do_sample) else '(disabled)'}")
    print(f"top_p:           {top_p if (not use_dfs and do_sample) else '(disabled)'}")
    print(f"pass_guess:      {pass_guess}")
    print(f"aug scoring:     {'enabled' if use_aug_score else 'disabled'}")
    print(f"input aug n:     {input_aug_n}")
    print(f"TTT train aug n: {ttt_aug_n}")
    print(f"TTT lr:          {ttt_learning_rate}")
    print(f"TTT emb lr:      {ttt_embedding_learning_rate}")
    print(f"TTT epochs:      {ttt_num_epochs}")
    print("=" * 80)

    eval_dataset = load_eval_dataset(arc_data_path, eval_task_limit)
    tokenizer_model, tokenizer = load_unsloth_model(str(base_model), bits=bits, device_map=get_single_device_map())
    sync_generation_special_tokens(tokenizer_model, tokenizer)
    fmt_opts = build_fmt_opts(tokenizer)
    del tokenizer_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_tool = EvalTool(n_guesses=2) if use_aug_score else make_eval_tool(use_aug_score=False, n_guesses=2)
    inference_results = {}
    inference_keys = {}
    completed_base_keys = set()

    results_file = output_path / "results.pickle.bz2"
    if results_file.is_file():
        try:
            with bz2.BZ2File(results_file, "rb") as handle:
                saved_inference_keys = pickle.load(handle)
                saved_inference_results = pickle.load(handle)
            if isinstance(saved_inference_keys, dict):
                inference_keys.update(saved_inference_keys)
            if isinstance(saved_inference_results, dict):
                inference_results.update(saved_inference_results)
                completed_base_keys.update(saved_inference_results.keys())
            print(f"Loaded existing aggregate results from: {results_file}")
        except Exception as exc:
            print(f"Could not load existing aggregate results from {results_file}: {exc}")

    for cached_state_path in sorted(task_state_dir.glob("*.pkl.bz2")):
        try:
            payload = load_task_state(cached_state_path)
            if not isinstance(payload, dict):
                continue
            saved_inference_keys = payload.get("inference_keys", {})
            saved_inference_results = payload.get("inference_results", {})
            if isinstance(saved_inference_keys, dict):
                inference_keys.update(saved_inference_keys)
            if isinstance(saved_inference_results, dict):
                inference_results.update(saved_inference_results)
                completed_base_keys.update(saved_inference_results.keys())
        except Exception as exc:
            print(f"Skipping unreadable task cache '{cached_state_path}': {exc}")

    split_parts = list(eval_dataset.split(n=len(eval_dataset.challenge), split_seed=123))

    with tqdm(list(enumerate(split_parts)), desc="direct+ttt inference") as pbar:
        for global_i, eval_dataset_part in pbar:
            base_key = sorted(eval_dataset_part.challenge.keys())[0]
            if base_key in completed_base_keys:
                pbar.write(f"[Direct+TTT] skipping completed task: {base_key}")
                continue

            model_tok_cache = [None]

            def get_model_and_tokenizer(cache=model_tok_cache):
                if cache[0] is not None:
                    return cache[0]

                model, task_tokenizer = load_unsloth_model(
                    str(base_model),
                    bits=bits,
                    device_map=get_single_device_map(),
                )
                sync_generation_special_tokens(model, task_tokenizer)
                model = FastLanguageModel.get_peft_model(
                    model=model,
                    target_modules=LORA_LAYERS,
                    r=32,
                    lora_alpha=16,
                    lora_dropout=0.0,
                    bias="none",
                    use_gradient_checkpointing=True,
                    random_state=42,
                    use_rslora=True,
                    loftq_config=None,
                )

                if lora_adapter_path is not None:
                    if not lora_adapter_path.is_dir():
                        raise FileNotFoundError(f"LORA_ADAPTER_PATH does not exist: {lora_adapter_path}")
                    load_peft_state(model, str(lora_adapter_path))
                    pbar.write(f"[Direct+TTT] loaded initial adapter: {lora_adapter_path}")

                train_dataset = eval_dataset_part.remove_test_data()
                if ttt_aug_n > 0:
                    train_aug_opts = dict(tp="all", rt="all", shfl_keys=True, perm=True, shfl_ex=True, seed=global_i)
                    train_dataset = train_dataset.augment(n=ttt_aug_n, **train_aug_opts)
                train_dataset_as_list = train_dataset.as_list(len_name="text", **fmt_opts)
                pbar.write(f"[Direct+TTT] task={base_key} train_rows={len(train_dataset_as_list)}")

                if train_dataset_as_list:
                    effective_warmup_steps = 0
                    if ttt_warmup_steps > 0:
                        effective_warmup_steps = min(ttt_warmup_steps, max(1, len(train_dataset_as_list)))

                    FastLanguageModel.for_training(model)
                    trainer = Trainer(
                        model=model,
                        tokenizer=task_tokenizer,
                        train_dataset=Dataset.from_list(train_dataset_as_list),
                        dataset_text_field="text",
                        max_seq_length=fmt_opts["max_tokens"],
                        data_collator=InputMaskingDataCollator(
                            instruction_template=fmt_opts["query_beg"],
                            response_template=fmt_opts["reply_beg"],
                            mlm=False,
                            tokenizer=task_tokenizer,
                            mask_first_n_examples=1,
                        ),
                        args=TrainingArguments(
                            per_device_train_batch_size=1,
                            gradient_accumulation_steps=1,
                            warmup_steps=effective_warmup_steps,
                            num_train_epochs=ttt_num_epochs,
                            learning_rate=ttt_learning_rate,
                            embedding_learning_rate=ttt_embedding_learning_rate,
                            fp16=not is_bfloat16_supported(),
                            bf16=is_bfloat16_supported(),
                            logging_steps=8,
                            optim="adamw_8bit",
                            weight_decay=0.00,
                            lr_scheduler_type="cosine",
                            seed=42,
                            output_dir="tmp_output",
                            save_strategy="no",
                            report_to="none",
                        ),
                    )
                    unsloth_train(trainer)

                FastLanguageModel.for_inference(model)
                disable_gradient_checkpointing(model)
                if hasattr(model, "generation_config"):
                    model.generation_config.use_cache = True
                cache[0] = (model, task_tokenizer)
                return cache[0]

            infer_aug_opts = dict(tp="all", rt="all", perm=True, shfl_ex=True, seed=10000 + global_i)
            inference_dataset = (
                eval_dataset_part.augment(n=input_aug_n, **infer_aug_opts) if input_aug_n > 1 else eval_dataset_part
            )
            for grouped_base_key, grouped_task_groups in inference_dataset.grouped_keys().items():
                inference_keys[grouped_base_key] = [key for group in grouped_task_groups for key in group]

            cache_decorator = None
            if use_cache:
                cache_decorator = Cache(str(output_path / f"{base_key}.cache")).memoize(
                    typed=True,
                    ignore=set(["model_tok", "guess"]),
                )

            aug_score_opts = None
            if use_aug_score:
                aug_score_opts = dict(**infer_aug_opts, n=2)

            inference_kwargs = dict(pass_guess=pass_guess)
            if use_dfs:
                inference_kwargs["min_prob"] = min_prob
            else:
                inference_kwargs["num_return_sequences"] = num_return_sequences
                inference_kwargs["do_sample"] = do_sample
                if do_sample:
                    inference_kwargs["temperature"] = temperature
                    inference_kwargs["top_p"] = top_p

            part_results = inference_run(
                model_tok=get_model_and_tokenizer,
                fmt_opts=fmt_opts,
                dataset=inference_dataset,
                aug_score_opts=aug_score_opts,
                callback=eval_tool.process_result,
                cache=cache_decorator,
                print_func=pbar.write,
                **inference_kwargs,
            )
            inference_results.update(part_results)
            save_task_state(task_state_dir / f"{base_key}.pkl.bz2", {base_key: inference_keys[base_key]}, part_results)
            completed_base_keys.add(base_key)

            cleanup_model(model_tok_cache[0])
            model_tok_cache[0] = None

    write_outputs(output_path, eval_dataset, inference_keys, inference_results)

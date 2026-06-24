#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    from .arc_loader import ArcDataset
    from .negative_transforms import (
        CLIPCalculator,
        build_instruction,
        generate_16_augmentations,
        generate_all_transforms,
        grid_to_string,
        process_grid_transforms,
    )
    from .paths import DATA_DIR
except ImportError:  # pragma: no cover
    from arc_loader import ArcDataset
    from negative_transforms import (
        CLIPCalculator,
        build_instruction,
        generate_16_augmentations,
        generate_all_transforms,
        grid_to_string,
        process_grid_transforms,
    )
    from paths import DATA_DIR

"""Build DPO preferences for ARC-style datasets without RE-ARC generators."""


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

DATASET_CHOICES = ("arcagi2", "conceptarc", "miniarc", "1d-arc", "arccommunity")
TRANSFORM_CHOICES = (
    "all",
    "grid_block",
    "rigid_shift",
    "morphology",
    "random_perturb",
    "rigid_shift_1d",
    "random_perturb_1d",
)
PROMPT_FORMAT_CHOICES = ("classic", "qwen")
RANKER_CHOICES = ("auto", "clip", "grid")


def stable_seed(*parts: object) -> int:
    text = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def default_output_dir(dataset: str, transform_category: str, arcagi2_split: str) -> Path:
    if dataset == "arcagi2":
        if transform_category == "all":
            subdir = f"dpo_arcagi2_{arcagi2_split}_transforms_16"
        else:
            subdir = f"dpo_arcagi2_{arcagi2_split}_{transform_category}_16"
    else:
        if transform_category == "all":
            subdir = f"dpo_{dataset}_transforms_16"
        else:
            subdir = f"dpo_{dataset}_{transform_category}_16"
    return DATA_DIR / subdir


def merge_task_with_solutions(task: dict, solutions: list | None) -> dict:
    merged = {"train": [], "test": []}
    for example in task.get("train", []):
        merged["train"].append({"input": example["input"], "output": example.get("output")})
    for index, example in enumerate(task.get("test", [])):
        merged_test = {"input": example["input"]}
        if "output" in example:
            merged_test["output"] = example["output"]
        elif solutions is not None and index < len(solutions):
            merged_test["output"] = solutions[index]
        else:
            merged_test["output"] = None
        merged["test"].append(merged_test)
    return merged


def iter_latest_json_files(data_root: Path):
    versioned: dict[str, tuple[int, Path]] = {}
    for file_path in data_root.rglob("*.json"):
        rel = file_path.relative_to(data_root).as_posix()
        match = re.match(r"^(?P<stem>.+)_v(?P<ver>\d+)\.json$", rel)
        if match:
            canonical_key = match.group("stem") + ".json"
            version = int(match.group("ver"))
        else:
            canonical_key = rel
            version = 0
        best = versioned.get(canonical_key)
        if best is None or version > best[0]:
            versioned[canonical_key] = (version, file_path)
    for canonical_key in sorted(versioned):
        yield versioned[canonical_key][1]


def load_arcagi2_tasks(dataset_root: Path, split: str) -> list[dict]:
    challenges_file = dataset_root / f"arc-agi_{split}_challenges.json"
    solutions_file = dataset_root / f"arc-agi_{split}_solutions.json"
    if not challenges_file.is_file():
        raise FileNotFoundError(f"missing ARC-AGI-2 challenges file: {challenges_file}")
    if not solutions_file.is_file():
        raise FileNotFoundError(f"missing ARC-AGI-2 solutions file: {solutions_file}")

    challenges = json.loads(challenges_file.read_text(encoding="utf-8"))
    solutions = json.loads(solutions_file.read_text(encoding="utf-8"))
    items = []
    for task_id in sorted(challenges):
        items.append(
            {
                "task_id": task_id,
                "task": merge_task_with_solutions(challenges[task_id], solutions.get(task_id)),
            }
        )
    return items


def load_neoneye_tasks(dataset_root: Path) -> list[dict]:
    data_root = dataset_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"missing dataset data dir: {data_root}")

    items = []
    for file_path in iter_latest_json_files(data_root):
        task = json.loads(file_path.read_text(encoding="utf-8"))
        rel = file_path.relative_to(data_root).with_suffix("")
        items.append({"task_id": rel.as_posix(), "task": merge_task_with_solutions(task, None)})
    return items


def load_tasks(dataset: str, dataset_root: Path, arcagi2_split: str) -> list[dict]:
    if dataset == "arcagi2":
        return load_arcagi2_tasks(dataset_root, arcagi2_split)
    if dataset in {"conceptarc", "miniarc", "1d-arc", "arccommunity"}:
        return load_neoneye_tasks(dataset_root)
    raise ValueError(f"unsupported dataset: {dataset}")


def resolve_transform_category(dataset: str, transform_category: str) -> str:
    if dataset != "1d-arc":
        if transform_category in {"rigid_shift_1d", "random_perturb_1d"}:
            raise ValueError(f"{transform_category} is only supported for dataset='1d-arc'")
        return transform_category

    one_dimensional = {
        "all": "all_1d",
        "rigid_shift": "rigid_shift_1d",
        "random_perturb": "random_perturb_1d",
        "rigid_shift_1d": "rigid_shift_1d",
        "random_perturb_1d": "random_perturb_1d",
    }
    if transform_category not in one_dimensional:
        raise ValueError(
            "1D-ARC supports only all, rigid_shift, random_perturb, "
            "rigid_shift_1d, or random_perturb_1d transforms"
        )
    return one_dimensional[transform_category]


def generate_transforms_for_dataset(dataset: str, transform_category: str):
    resolved = resolve_transform_category(dataset, transform_category)
    if resolved == "all_1d":
        return (
            generate_all_transforms(category_filter="rigid_shift_1d")
            + generate_all_transforms(category_filter="random_perturb_1d")
        )
    return generate_all_transforms(category_filter=resolved)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_qwen_instruction(
    train_data,
    test_input,
    d8_transforms=None,
    perm_str=None,
    shuffle_order=None,
):
    if shuffle_order is not None:
        ordered_train_data = [train_data[i] for i in shuffle_order if i < len(train_data)]
    else:
        ordered_train_data = train_data

    transforms = []
    if d8_transforms:
        transforms.extend(d8_transforms)
    if perm_str:
        transforms.append("perm" + perm_str)

    chunks = []
    for train_ex in ordered_train_data:
        input_grid = np.asarray(train_ex["input"])
        output_grid = np.asarray(train_ex["output"])

        if transforms:
            input_grid = ArcDataset.transform_array(input_grid, transforms)
            output_grid = ArcDataset.transform_array(output_grid, transforms)

        input_str = grid_to_string(input_grid)
        output_str = grid_to_string(output_grid)
        chunks.append(
            "<|im_start|>user\n"
            + input_str
            + "<|im_end|><|im_start|>assistant\n"
            + output_str
            + "<|im_end|>"
        )

    test_input_grid = np.asarray(test_input)
    if transforms:
        test_input_grid = ArcDataset.transform_array(test_input_grid, transforms)
    test_input_str = grid_to_string(test_input_grid)
    chunks.append("<|im_start|>user\n" + test_input_str + "<|im_end|><|im_start|>assistant\n")
    return "".join(chunks), transforms


def build_case_samples(
    *,
    dataset: str,
    task_id: str,
    test_index: int,
    task: dict,
    transforms: list[tuple[str, list[str], str]],
    clip_calc: CLIPCalculator,
    fmt_opts: dict,
    augmentation_list: list[dict],
    top_k: int,
    no_augment: bool,
    source_split: str | None,
    prompt_format: str,
) -> list[dict]:
    train_examples = task.get("train", [])
    test_examples = task.get("test", [])
    if test_index >= len(test_examples):
        return []

    test_example = test_examples[test_index]
    if test_example.get("output") is None:
        return []

    test_input = test_example["input"]
    chosen_output = np.array(test_example["output"])
    color_candidates = set(np.unique(chosen_output).tolist())
    results = process_grid_transforms(chosen_output, transforms, clip_calc, color_candidates)
    if not results:
        return []

    ranked_results = [row for row in results if abs(row["similarity"] - 1.0) >= 1e-6][:top_k]
    if not ranked_results:
        return []

    samples = []
    source_info = f"{dataset}:{task_id}[test={test_index}]"
    case_key = f"{task_id}__test{test_index}"
    for rank, result in enumerate(ranked_results):
        if no_augment:
            if prompt_format == "qwen":
                instruction, _ = build_qwen_instruction(train_examples, test_input)
            else:
                instruction, _ = build_instruction(train_examples, test_input, fmt_opts)
            chosen_grid = chosen_output
            rejected_grid = result["grid"]
            applied_aug = []
            shuffle_order = None
        else:
            sample_seed = stable_seed(dataset, task_id, test_index, rank)
            np.random.seed(sample_seed)
            random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)

            aug_idx = np.random.randint(len(augmentation_list))
            aug_config = augmentation_list[aug_idx]
            perm_str = ArcDataset.rand_perm(10, "", keep_zero=False)
            shuffle_order = (
                np.random.permutation(len(train_examples)).tolist() if aug_config["shuffle"] else None
            )
            if prompt_format == "qwen":
                instruction, applied_aug = build_qwen_instruction(
                    train_examples,
                    test_input,
                    d8_transforms=aug_config["d8"],
                    perm_str=perm_str,
                    shuffle_order=shuffle_order,
                )
            else:
                instruction, applied_aug = build_instruction(
                    train_examples,
                    test_input,
                    fmt_opts,
                    d8_transforms=aug_config["d8"],
                    perm_str=perm_str,
                    shuffle_order=shuffle_order,
                )
            chosen_grid = ArcDataset.transform_array(chosen_output, applied_aug)
            rejected_grid = ArcDataset.transform_array(result["grid"], applied_aug)

        chosen_text = grid_to_string(chosen_grid)
        rejected_text = grid_to_string(rejected_grid)
        if prompt_format == "qwen":
            chosen_text += "<|im_end|>"
            rejected_text += "<|im_end|>"

        sample = {
            "instruction": instruction,
            "input": "",
            "chosen": chosen_text,
            "rejected": rejected_text,
            "rank": rank,
            "combo_type": result.get("combo_type", "unknown"),
            "similarity": result["similarity"],
            "task_id": case_key,
            "source": source_info,
            "mode": "external_output_transform",
            "prompt_format": prompt_format,
            "source_dataset": dataset,
            "source_task_id": task_id,
            "source_test_index": test_index,
            "source_split": source_split,
            "transform_methods": result.get("methods", []),
        }
        if not no_augment:
            sample["augmentation_transforms"] = applied_aug
            sample["augmentation_shuffle_order"] = shuffle_order
        samples.append(sample)

    return samples


def build_dataset(
    *,
    dataset: str,
    dataset_root: Path,
    output_dir: Path | None = None,
    transform_category: str = "all",
    top_k: int = 16,
    no_augment: bool = False,
    seed: int = 42,
    max_tasks: int | None = None,
    arcagi2_split: str = "training",
    prompt_format: str = "classic",
    ranker: str = "auto",
) -> Path:
    if dataset not in DATASET_CHOICES:
        raise ValueError(f"unsupported dataset: {dataset}")
    if transform_category not in TRANSFORM_CHOICES:
        raise ValueError(f"unsupported transform category: {transform_category}")
    if prompt_format not in PROMPT_FORMAT_CHOICES:
        raise ValueError(f"unsupported prompt format: {prompt_format}")
    if ranker not in RANKER_CHOICES:
        raise ValueError(f"unsupported ranker: {ranker}")

    start_time = time.time()
    start_datetime = datetime.now()

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    source_split = arcagi2_split if dataset == "arcagi2" else None
    output_dir = (output_dir or default_output_dir(dataset, transform_category, arcagi2_split)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt_opts = dict(
        preprompt="ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz",
        query_beg="I",
        reply_beg="\n+/-=O",
        reply_end="\n</s>",
        lines_sep="\n",
    )

    transforms = generate_transforms_for_dataset(dataset, transform_category)
    augmentation_list = generate_16_augmentations()
    tasks = load_tasks(dataset, dataset_root, arcagi2_split)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    print(f"\n开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("配置:")
    print(f"  数据集: {dataset}")
    print(f"  数据根目录: {dataset_root}")
    if source_split is not None:
        print(f"  split: {source_split}")
    print(f"  输出目录: {output_dir}")
    print(f"  变换类别: {transform_category}")
    print(f"  变换总数: {len(transforms)}")
    print(f"  top_k: {top_k}")
    print(f"  增广: {'关闭' if no_augment else '16种'}")
    print(f"  prompt格式: {prompt_format}")
    print(f"  ranker: {ranker}")
    print(f"  随机种子: {seed}")
    print("=" * 60)

    clip_calc = None
    actual_ranker = "grid"
    if ranker in {"auto", "clip"}:
        try:
            clip_calc = CLIPCalculator(device="cuda" if torch.cuda.is_available() else "cpu")
            actual_ranker = "clip"
        except Exception as exc:
            if ranker == "clip":
                raise
            print(f"CLIP ranker unavailable ({exc!r}); falling back to grid similarity.")

    all_samples: list[dict] = []
    stats = {
        "loaded_tasks": len(tasks),
        "processed_cases": 0,
        "skipped_cases": 0,
        "cases_with_less_than_k": 0,
        "task_ids_with_output": set(),
    }

    for item in tqdm(tasks, desc=f"Processing {dataset}"):
        task_id = item["task_id"]
        task = item["task"]
        for test_index, test_example in enumerate(task.get("test", [])):
            if test_example.get("output") is None:
                stats["skipped_cases"] += 1
                continue

            samples = build_case_samples(
                dataset=dataset,
                task_id=task_id,
                test_index=test_index,
                task=task,
                transforms=transforms,
                clip_calc=clip_calc,
                fmt_opts=fmt_opts,
                augmentation_list=augmentation_list,
                top_k=top_k,
                no_augment=no_augment,
                source_split=source_split,
                prompt_format=prompt_format,
            )
            if samples:
                stats["processed_cases"] += 1
                stats["task_ids_with_output"].add(task_id)
                if len(samples) < top_k:
                    stats["cases_with_less_than_k"] += 1
                all_samples.extend(samples)
            else:
                stats["skipped_cases"] += 1

    stage_counts: dict[int, int] = {}
    stage_size = 4
    num_stages = max(1, math.ceil(top_k / stage_size))
    print(f"\nGenerating {num_stages} stage files...")
    for stage in range(1, num_stages + 1):
        stage_ranks = list(range((stage - 1) * stage_size, min(stage * stage_size, top_k)))
        stage_samples = [sample for sample in all_samples if sample["rank"] in stage_ranks]
        stage_counts[stage] = len(stage_samples)
        write_jsonl(output_dir / f"arc_dpo_data_stage_{stage}.jsonl", stage_samples)
        print(f"  Stage {stage} (rank {stage_ranks}): {len(stage_samples)} samples")

    write_jsonl(output_dir / "arc_dpo_data_all.jsonl", all_samples)

    end_time = time.time()
    end_datetime = datetime.now()
    elapsed_seconds = end_time - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed_seconds)))
    task_ids_with_output = sorted(stats["task_ids_with_output"])
    del stats["task_ids_with_output"]

    stats["task_count_with_output"] = len(task_ids_with_output)
    stats["total_samples"] = len(all_samples)
    stats["stage_counts"] = {str(k): v for k, v in stage_counts.items()}

    generation_info = {
        "dataset": dataset,
        "dataset_root": str(dataset_root),
        "split": source_split,
        "transform_category": transform_category,
        "top_k": top_k,
        "no_augment": no_augment,
        "augmentation": "none" if no_augment else "16种 (8 D8 × 2 示例顺序 + 颜色置换)",
        "prompt_format": prompt_format,
        "ranker": actual_ranker,
        "clip_model": "ViT-L/14@336px" if actual_ranker == "clip" else None,
        "mode": "external_output_transform",
        "description": "chosen 取自外部数据集 task 的 gold test output，rejected 为 chosen_output 的变换结果",
        "loaded_tasks": stats["loaded_tasks"],
        "processed_cases": stats["processed_cases"],
        "skipped_cases": stats["skipped_cases"],
        "cases_with_less_than_k": stats["cases_with_less_than_k"],
        "task_count_with_output": stats["task_count_with_output"],
        "total_samples": stats["total_samples"],
        "stage_counts": stats["stage_counts"],
        "seed": seed,
        "start_time": start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed_seconds,
    }
    (output_dir / "generation_info.json").write_text(
        json.dumps(generation_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print("数据生成完成!")
    print(f"{'=' * 60}")
    print(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总用时: {elapsed_str}")
    print(f"Loaded tasks: {stats['loaded_tasks']}")
    print(f"Processed cases: {stats['processed_cases']}")
    print(f"Skipped cases: {stats['skipped_cases']}")
    print(f"Cases with < {top_k} samples: {stats['cases_with_less_than_k']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO data by transforming gold outputs from external ARC-style datasets.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--transform-category", choices=TRANSFORM_CHOICES, default="all")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--arcagi2-split", choices=["training", "evaluation"], default="training")
    parser.add_argument("--prompt-format", choices=PROMPT_FORMAT_CHOICES, default="classic")
    parser.add_argument("--ranker", choices=RANKER_CHOICES, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        transform_category=args.transform_category,
        top_k=args.top_k,
        no_augment=args.no_augment,
        seed=args.seed,
        max_tasks=args.max_tasks,
        arcagi2_split=args.arcagi2_split,
        prompt_format=args.prompt_format,
        ranker=args.ranker,
    )


if __name__ == "__main__":
    main()

import json
import re
import sys
from pathlib import Path


try:
    from .paths import DATA_DIR, RE_ARC_GEN_DIR
except ImportError:  # pragma: no cover
    from paths import DATA_DIR, RE_ARC_GEN_DIR

if str(RE_ARC_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(RE_ARC_GEN_DIR))


FMT_OPTS = {
    "preprompt": "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz",
    "query_beg": "I",
    "reply_beg": "\n+/-=O",
    "reply_end": "\n</s>",
    "lines_sep": "\n",
}

_VERIFIERS_MODULE = None


def get_verifiers_module():
    global _VERIFIERS_MODULE
    if _VERIFIERS_MODULE is None:
        import verifiers  # noqa: E402

        _VERIFIERS_MODULE = verifiers
    return _VERIFIERS_MODULE


def is_grid(grid) -> bool:
    if not isinstance(grid, tuple):
        return False
    if not 0 < len(grid) <= 30:
        return False
    if not all(isinstance(row, tuple) for row in grid):
        return False
    if not all(0 < len(row) <= 30 for row in grid):
        return False
    if len({len(row) for row in grid}) != 1:
        return False
    if not all(all(isinstance(value, int) for value in row) for row in grid):
        return False
    if not all(all(0 <= value <= 9 for value in row) for row in grid):
        return False
    return True


def list_grid_to_tuple(grid):
    return tuple(tuple(row) for row in grid)


def grid_to_string(grid):
    return "\n".join("".join(map(str, row)) for row in grid)


def grid_key(grid):
    return json.dumps(grid, separators=(",", ":"))


def get_task_ids_in_file_order(task_limit: int) -> list[str]:
    source = (RE_ARC_GEN_DIR / "generators.py").read_text(encoding="utf-8")
    task_ids = re.findall(r"def generate_([0-9a-f]+)\(", source)
    return task_ids[:task_limit]


def build_instruction(train_examples, test_input):
    prefix = FMT_OPTS["preprompt"]
    query_beg = FMT_OPTS["query_beg"]
    reply_beg = FMT_OPTS["reply_beg"]
    reply_end = FMT_OPTS["reply_end"]
    lines_sep = FMT_OPTS["lines_sep"]

    examples = []
    for example in train_examples:
        input_str = lines_sep.join("".join(map(str, row)) for row in example["input"])
        output_str = lines_sep.join("".join(map(str, row)) for row in example["output"])
        examples.append(f"{query_beg}{input_str}{reply_beg}{output_str}{reply_end}")

    test_input_str = lines_sep.join("".join(map(str, row)) for row in test_input)
    return prefix + "".join(examples) + f"{query_beg}{test_input_str}{reply_beg}"


def get_verifier(task_id):
    return getattr(get_verifiers_module(), f"verify_{task_id}")


def build_dpo_dataset_jsonl(
    output_path: Path,
    original_rearc_root: Path,
    negative_root: Path,
    task_limit: int,
    n_shots: int = 6,
    required_examples_per_task: int | None = None,
):
    original_tasks_dir = original_rearc_root / "tasks"
    negative_tasks_dir = negative_root / "tasks"
    selected_task_ids = get_task_ids_in_file_order(task_limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    skipped = 0
    chosen_source_stats = {
        "verifier_from_negative_input": 0,
        "skipped_unpairable": 0,
        "skipped_same_as_rejected": 0,
        "skipped_bad_example_count": 0,
    }
    with output_path.open("w", encoding="utf-8") as out_fp:
        for task_id in selected_task_ids:
            task_file = negative_tasks_dir / f"{task_id}.json"
            if not task_file.exists():
                raise RuntimeError(f"missing negative task file: {task_file}")
            original_task_file = original_tasks_dir / f"{task_id}.json"
            if not original_task_file.exists():
                raise RuntimeError(f"missing original rearc task file: {original_task_file}")

            original_examples = json.loads(original_task_file.read_text())
            negative_examples = json.loads(task_file.read_text())
            verifier = get_verifier(task_id)

            if required_examples_per_task is not None and len(negative_examples) != required_examples_per_task:
                skipped += 1
                chosen_source_stats["skipped_bad_example_count"] += 1
                continue

            if len(negative_examples) < n_shots + 1:
                skipped += 1
                chosen_source_stats["skipped_unpairable"] += 1
                continue

            group_size = n_shots + 1
            n_groups = len(negative_examples) // group_size

            for group_idx in range(n_groups):
                group = negative_examples[group_idx * group_size:(group_idx + 1) * group_size]
                support_negative = group[:n_shots]
                query_negative = group[n_shots]

                support_examples = []
                support_ok = True
                for support_idx, support_ex in enumerate(support_negative):
                    try:
                        support_output = verifier(list_grid_to_tuple(support_ex["input"]))
                    except Exception:
                        support_ok = False
                        break
                    if not is_grid(support_output):
                        support_ok = False
                        break
                    support_examples.append(
                        {
                            "input": support_ex["input"],
                            "output": [list(row) for row in support_output],
                        }
                    )

                if not support_ok:
                    skipped += 1
                    chosen_source_stats["skipped_unpairable"] += 1
                    continue

                prompt_input = query_negative["input"]
                try:
                    chosen_grid = verifier(list_grid_to_tuple(prompt_input))
                    chosen_source = "verifier_from_negative_input"
                except Exception:
                    skipped += 1
                    chosen_source_stats["skipped_unpairable"] += 1
                    continue

                rejected_grid = list_grid_to_tuple(query_negative["output"])
                if not (is_grid(chosen_grid) and is_grid(rejected_grid)):
                    skipped += 1
                    chosen_source_stats["skipped_unpairable"] += 1
                    continue
                if chosen_grid == rejected_grid:
                    skipped += 1
                    chosen_source_stats["skipped_same_as_rejected"] += 1
                    continue

                sample = {
                    "instruction": build_instruction(support_examples, prompt_input),
                    "input": "",
                    "chosen": grid_to_string(chosen_grid),
                    "rejected": grid_to_string(rejected_grid),
                    "task_id": task_id,
                    "chosen_source": chosen_source,
                    "negative_source": negative_root.name,
                    "negative_group": group_idx,
                }
                out_fp.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_written += 1
                chosen_source_stats[chosen_source] += 1

    return {
        "total_written": total_written,
        "skipped": skipped,
        "chosen_source_stats": chosen_source_stats,
    }


def run_build(
    negative_name: str,
    task_limit: int,
    dataset_subdir: str,
    required_examples_per_task: int | None = 20,
):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    original_rearc_root = DATA_DIR / "re_arc"
    negative_root = RE_ARC_GEN_DIR / negative_name
    dataset_dir = DATA_DIR / dataset_subdir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else dataset_dir / "arc_dpo_data_all.jsonl"

    stats = build_dpo_dataset_jsonl(
        output_path=output_path,
        original_rearc_root=original_rearc_root,
        negative_root=negative_root,
        task_limit=task_limit,
        required_examples_per_task=required_examples_per_task,
    )
    print(f"built {stats['total_written']} samples -> {output_path}")
    print(f"skipped {stats['skipped']} unpairable negatives")
    print(f"chosen source stats: {stats['chosen_source_stats']}")

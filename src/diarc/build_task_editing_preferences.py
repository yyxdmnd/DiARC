from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from .paths import DATA_DIR, RE_ARC_GEN_DIR
    from .rearc_program_utils import (
        build_instruction,
        get_task_ids_in_file_order,
        grid_key,
        load_module,
        load_rearc_runtime,
        sample_edited_query,
        sample_support_examples,
    )
except ImportError:  # pragma: no cover
    from paths import DATA_DIR, RE_ARC_GEN_DIR
    from rearc_program_utils import (
        build_instruction,
        get_task_ids_in_file_order,
        grid_key,
        load_module,
        load_rearc_runtime,
        sample_edited_query,
        sample_support_examples,
    )


def build_dataset(
    *,
    rearc_root: Path,
    output_path: Path,
    edited_verifier_module_path: Path | None,
    edited_generator_module_path: Path | None,
    task_limit: int | None,
    task_ids: set[str] | None,
    groups_per_task: int,
    n_shots: int,
    diff_lb: float,
    diff_ub: float,
    max_attempts: int,
) -> dict:
    if (edited_verifier_module_path is None) == (edited_generator_module_path is None):
        raise ValueError("pass exactly one of --edited-verifier-module or --edited-generator-module")

    _, generators_module, verifiers_module = load_rearc_runtime(rearc_root)
    edited_module = load_module(
        edited_verifier_module_path or edited_generator_module_path,
        "diarc_task_specific_edits",
    )
    edited_fn_is_generator = edited_generator_module_path is not None

    ordered_task_ids = get_task_ids_in_file_order(rearc_root, task_limit)
    if task_ids is not None:
        ordered_task_ids = [task_id for task_id in ordered_task_ids if task_id in task_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with output_path.open("w", encoding="utf-8") as out_fp:
        for task_id in ordered_task_ids:
            generator_fn = getattr(generators_module, f"generate_{task_id}", None)
            original_verifier = getattr(verifiers_module, f"verify_{task_id}", None)
            edited_name = f"generate_{task_id}" if edited_fn_is_generator else f"verify_{task_id}"
            edited_fn = getattr(edited_module, edited_name, None)
            if generator_fn is None or original_verifier is None or edited_fn is None:
                stats["missing_runtime_function"] += 1
                continue

            for group_index in range(groups_per_task):
                support = sample_support_examples(
                    generator_fn=generator_fn,
                    original_verifier=original_verifier,
                    count=n_shots,
                    diff_lb=diff_lb,
                    diff_ub=diff_ub,
                    max_attempts=max_attempts,
                )
                if len(support) < n_shots:
                    stats["insufficient_support"] += 1
                    continue
                excluded = {grid_key(row["input"]) for row in support}
                query = sample_edited_query(
                    generator_fn=generator_fn,
                    original_verifier=original_verifier,
                    edited_fn=edited_fn,
                    diff_lb=diff_lb,
                    diff_ub=diff_ub,
                    max_attempts=max_attempts,
                    edited_fn_is_generator=edited_fn_is_generator,
                    excluded_input_keys=excluded,
                )
                if query is None:
                    stats["insufficient_query"] += 1
                    continue

                sample = {
                    "instruction": build_instruction(support, query["input"]),
                    "input": "",
                    "chosen": query["chosen"],
                    "rejected": query["rejected"],
                    "task_id": task_id,
                    "negative_method": "task_specific_editing",
                    "edit_source": "edited_generator" if edited_fn_is_generator else "edited_verifier",
                    "group_index": group_index,
                }
                out_fp.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["written"] += 1

    return {"output_path": str(output_path), "stats": dict(stats)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO data from task-specific edited ARC programs.")
    parser.add_argument("--rearc-root", type=Path, default=RE_ARC_GEN_DIR)
    parser.add_argument("--edited-verifier-module", type=Path, default=None)
    parser.add_argument("--edited-generator-module", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "dpo_arcagi1_task_editing" / "arc_dpo_data_all.jsonl")
    parser.add_argument("--task-limit", type=int, default=400)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--groups-per-task", type=int, default=2)
    parser.add_argument("--n-shots", type=int, default=6)
    parser.add_argument("--diff-lb", type=float, default=0.0)
    parser.add_argument("--diff-ub", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=4000)
    parser.add_argument("--metadata-output", type=Path, default=None)
    args = parser.parse_args()

    stats = build_dataset(
        rearc_root=args.rearc_root,
        output_path=args.output,
        edited_verifier_module_path=args.edited_verifier_module,
        edited_generator_module_path=args.edited_generator_module,
        task_limit=args.task_limit,
        task_ids=set(args.task_ids) if args.task_ids else None,
        groups_per_task=args.groups_per_task,
        n_shots=args.n_shots,
        diff_lb=args.diff_lb,
        diff_ub=args.diff_ub,
        max_attempts=args.max_attempts,
    )
    metadata_output = args.metadata_output or args.output.with_name("build_stats.json")
    metadata_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"built {stats['stats'].get('written', 0)} samples -> {args.output}")
    print(f"wrote stats -> {metadata_output}")
    print(f"stats: {stats['stats']}")


if __name__ == "__main__":
    main()


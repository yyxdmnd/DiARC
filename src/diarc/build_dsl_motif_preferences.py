from __future__ import annotations

import argparse
import ast
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .paths import DATA_DIR, RE_ARC_GEN_DIR
    from .rearc_program_utils import (
        build_instruction,
        get_task_ids_in_file_order,
        grid_key,
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
        load_rearc_runtime,
        sample_edited_query,
        sample_support_examples,
    )


class PaintToUnderpaint(ast.NodeTransformer):
    def __init__(self) -> None:
        self.edit_count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "paint" and len(node.args) == 2:
            node.func.id = "underpaint"
            self.edit_count += 1
        return node


class FillToUnderfill(ast.NodeTransformer):
    def __init__(self) -> None:
        self.edit_count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "fill" and len(node.args) == 3:
            node.func.id = "underfill"
            self.edit_count += 1
        return node


class OfcolorToMostcolor(ast.NodeTransformer):
    def __init__(self) -> None:
        self.edit_count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "ofcolor" and len(node.args) == 2:
            node.args[1] = ast.Call(func=ast.Name(id="mostcolor", ctx=ast.Load()), args=[node.args[0]], keywords=[])
            self.edit_count += 1
        return node


class ObjectsFlipWithoutBg(ast.NodeTransformer):
    def __init__(self) -> None:
        self.edit_count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "objects" and len(node.args) == 4:
            node.args[3] = ast.Call(func=ast.Name(id="flip", ctx=ast.Load()), args=[node.args[3]], keywords=[])
            self.edit_count += 1
        return node


class CanvasDimSwap(ast.NodeTransformer):
    def __init__(self) -> None:
        self.edit_count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "canvas" and len(node.args) == 2:
            dims = node.args[1]
            self.edit_count += 1
            return ast.Call(
                func=ast.Name(id="canvas", ctx=ast.Load()),
                args=[
                    node.args[0],
                    ast.Call(
                        func=ast.Name(id="astuple", ctx=ast.Load()),
                        args=[
                            ast.Call(func=ast.Name(id="last", ctx=ast.Load()), args=[dims], keywords=[]),
                            ast.Call(func=ast.Name(id="first", ctx=ast.Load()), args=[dims], keywords=[]),
                        ],
                        keywords=[],
                    ),
                ],
                keywords=[],
            )
        return node


class SfilterComplement(ast.NodeTransformer):
    def __init__(self) -> None:
        self.edit_count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "sfilter" and len(node.args) == 2:
            original = ast.Call(func=ast.Name(id="sfilter", ctx=ast.Load()), args=node.args, keywords=[])
            self.edit_count += 1
            return ast.Call(func=ast.Name(id="difference", ctx=ast.Load()), args=[node.args[0], original], keywords=[])
        return node


class SwapBranchArms(ast.NodeTransformer):
    def __init__(self) -> None:
        self.edit_count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "branch" and len(node.args) == 3:
            node.args = [node.args[0], node.args[2], node.args[1]]
            self.edit_count += 1
        return node


class FgpartitionToPartition(ast.NodeTransformer):
    def __init__(self) -> None:
        self.edit_count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "fgpartition" and len(node.args) == 1:
            node.func.id = "partition"
            self.edit_count += 1
        return node


class RemoveKeepOnlyRemovedColor(ast.NodeTransformer):
    def __init__(self) -> None:
        self.edit_count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "remove" and len(node.args) == 2:
            self.edit_count += 1
            return ast.Call(func=ast.Name(id="initset", ctx=ast.Load()), args=[node.args[0]], keywords=[])
        return node


class CombineToLeftOnly(ast.NodeTransformer):
    def __init__(self) -> None:
        self.edit_count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "combine" and len(node.args) == 2:
            self.edit_count += 1
            return node.args[0]
        return node


class MergeToFirst(ast.NodeTransformer):
    def __init__(self) -> None:
        self.edit_count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "merge" and len(node.args) == 1:
            node.func.id = "first"
            self.edit_count += 1
        return node


class SwapHeightWidth(ast.NodeTransformer):
    def __init__(self) -> None:
        self.edit_count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "height":
            node.func.id = "width"
            self.edit_count += 1
        elif isinstance(node.func, ast.Name) and node.func.id == "width":
            node.func.id = "height"
            self.edit_count += 1
        return node


REWRITE_FACTORIES = {
    "paint_to_underpaint": PaintToUnderpaint,
    "fill_to_underfill": FillToUnderfill,
    "ofcolor_to_mostcolor": OfcolorToMostcolor,
    "objects_flip_without_bg": ObjectsFlipWithoutBg,
    "canvas_dim_swap": CanvasDimSwap,
    "sfilter_complement": SfilterComplement,
    "branch_swap": SwapBranchArms,
    "fgpartition_to_partition": FgpartitionToPartition,
    "remove_keep_only_removed": RemoveKeepOnlyRemovedColor,
    "combine_left_only": CombineToLeftOnly,
    "merge_to_first": MergeToFirst,
    "height_width_swap": SwapHeightWidth,
}


def load_verifier_ast(verifiers_path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(verifiers_path.read_text(encoding="utf-8"))
    return {
        node.name.split("_", 1)[1]: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("verify_")
    }


def build_modified_verifier(func_def: ast.FunctionDef, transformer_factory, dsl_env: dict[str, Any]):
    transformer = transformer_factory()
    edited_def = copy.deepcopy(func_def)
    edited_def.name = f"{func_def.name}_edited"
    edited_def = transformer.visit(edited_def)
    ast.fix_missing_locations(edited_def)
    module = ast.Module(body=[edited_def], type_ignores=[])
    env = dict(dsl_env)
    exec(compile(module, filename="<diarc-dsl-rewrite>", mode="exec"), env)
    return env[edited_def.name], transformer.edit_count


def build_dataset(
    *,
    rearc_root: Path,
    output_path: Path,
    rewrites: list[str],
    task_limit: int | None,
    task_ids: set[str] | None,
    groups_per_task: int,
    n_shots: int,
    diff_lb: float,
    diff_ub: float,
    max_attempts: int,
) -> dict[str, Any]:
    dsl_module, generators_module, verifiers_module = load_rearc_runtime(rearc_root)
    dsl_env = {name: getattr(dsl_module, name) for name in dir(dsl_module) if not name.startswith("__")}
    verifier_ast = load_verifier_ast(Path(rearc_root) / "verifiers.py")
    ordered_task_ids = get_task_ids_in_file_order(rearc_root, task_limit)
    if task_ids is not None:
        ordered_task_ids = [task_id for task_id in ordered_task_ids if task_id in task_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with output_path.open("w", encoding="utf-8") as out_fp:
        for rewrite_name in rewrites:
            transformer_factory = REWRITE_FACTORIES[rewrite_name]
            for task_id in ordered_task_ids:
                generator_fn = getattr(generators_module, f"generate_{task_id}", None)
                original_verifier = getattr(verifiers_module, f"verify_{task_id}", None)
                func_def = verifier_ast.get(task_id)
                if generator_fn is None or original_verifier is None or func_def is None:
                    stats["missing_runtime_function"] += 1
                    continue

                try:
                    edited_verifier, edit_count = build_modified_verifier(func_def, transformer_factory, dsl_env)
                except Exception:
                    stats["compile_error"] += 1
                    continue
                if edit_count == 0:
                    stats["no_rewrite_match"] += 1
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
                        edited_fn=edited_verifier,
                        diff_lb=diff_lb,
                        diff_ub=diff_ub,
                        max_attempts=max_attempts,
                        edited_fn_is_generator=False,
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
                        "negative_method": "dsl_level_inversion",
                        "rewrite": rewrite_name,
                        "group_index": group_index,
                        "edit_count": edit_count,
                    }
                    out_fp.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    stats["written"] += 1

    return {"output_path": str(output_path), "stats": dict(stats)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO data with reusable DSL-level rule inversions.")
    parser.add_argument("--rearc-root", type=Path, default=RE_ARC_GEN_DIR)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "dpo_arcagi1_dsl_motif" / "arc_dpo_data_all.jsonl")
    parser.add_argument("--rewrite", nargs="+", default=["all"], help="One or more rewrite names, or 'all'.")
    parser.add_argument("--task-limit", type=int, default=400)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--groups-per-task", type=int, default=2)
    parser.add_argument("--n-shots", type=int, default=6)
    parser.add_argument("--diff-lb", type=float, default=0.0)
    parser.add_argument("--diff-ub", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=4000)
    parser.add_argument("--metadata-output", type=Path, default=None)
    args = parser.parse_args()

    rewrites = list(REWRITE_FACTORIES) if args.rewrite == ["all"] else args.rewrite
    unknown = sorted(set(rewrites) - set(REWRITE_FACTORIES))
    if unknown:
        raise ValueError(f"unknown rewrite names: {unknown}")

    stats = build_dataset(
        rearc_root=args.rearc_root,
        output_path=args.output,
        rewrites=rewrites,
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


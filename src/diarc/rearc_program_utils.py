from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable


FMT_OPTS = {
    "preprompt": "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz",
    "query_beg": "I",
    "reply_beg": "\n+/-=O",
    "reply_end": "\n</s>",
    "lines_sep": "\n",
}


def load_module(path: Path, module_name: str):
    path = Path(path).resolve()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def list_grid_to_tuple(grid) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid)


def tuple_grid_to_list(grid) -> list[list[int]]:
    return [list(row) for row in grid]


def is_grid(grid: Any) -> bool:
    if isinstance(grid, list):
        try:
            grid = list_grid_to_tuple(grid)
        except Exception:
            return False
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
    return all(all(0 <= value <= 9 for value in row) for row in grid)


def grid_to_string(grid) -> str:
    return "\n".join("".join(str(value) for value in row) for row in grid)


def grid_key(grid) -> str:
    return json.dumps(tuple_grid_to_list(list_grid_to_tuple(grid)), separators=(",", ":"))


def stable_seed(*parts: str) -> int:
    payload = "::".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha1(payload).digest()[:8], "big")


def build_instruction(train_examples: list[dict[str, Any]], test_input) -> str:
    chunks = []
    for example in train_examples:
        input_str = FMT_OPTS["lines_sep"].join("".join(map(str, row)) for row in example["input"])
        output_str = FMT_OPTS["lines_sep"].join("".join(map(str, row)) for row in example["output"])
        chunks.append(f"{FMT_OPTS['query_beg']}{input_str}{FMT_OPTS['reply_beg']}{output_str}{FMT_OPTS['reply_end']}")

    test_input_str = FMT_OPTS["lines_sep"].join("".join(map(str, row)) for row in test_input)
    return FMT_OPTS["preprompt"] + "".join(chunks) + f"{FMT_OPTS['query_beg']}{test_input_str}{FMT_OPTS['reply_beg']}"


def get_task_ids_in_file_order(rearc_root: Path, task_limit: int | None = None) -> list[str]:
    source_path = Path(rearc_root) / "generators.py"
    source = source_path.read_text(encoding="utf-8")
    task_ids = re.findall(r"def generate_([0-9a-f]+)\(", source)
    return task_ids[:task_limit] if task_limit else task_ids


def load_rearc_runtime(rearc_root: Path):
    rearc_root = Path(rearc_root).resolve()
    if str(rearc_root) not in sys.path:
        sys.path.insert(0, str(rearc_root))
    dsl_module = load_module(rearc_root / "dsl.py", "diarc_rearc_dsl")
    generators_module = load_module(rearc_root / "generators.py", "diarc_rearc_generators")
    verifiers_module = load_module(rearc_root / "verifiers.py", "diarc_rearc_verifiers")
    return dsl_module, generators_module, verifiers_module


def call_generator(generator_fn: Callable, diff_lb: float, diff_ub: float) -> dict[str, Any]:
    example = generator_fn(diff_lb, diff_ub)
    if not isinstance(example, dict) or "input" not in example:
        raise ValueError("RE-ARC generator must return a dict with an 'input' grid")
    return example


def sample_support_examples(
    *,
    generator_fn: Callable,
    original_verifier: Callable,
    count: int,
    diff_lb: float,
    diff_ub: float,
    max_attempts: int,
    excluded_input_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded_input_keys = excluded_input_keys or set()
    examples: list[dict[str, Any]] = []
    seen_inputs: set[str] = set(excluded_input_keys)
    attempts = 0

    while len(examples) < count and attempts < max_attempts:
        attempts += 1
        try:
            generated = call_generator(generator_fn, diff_lb, diff_ub)
            input_grid = list_grid_to_tuple(generated["input"])
            if not is_grid(input_grid):
                continue
            key = grid_key(input_grid)
            if key in seen_inputs:
                continue
            output_grid = original_verifier(input_grid)
            if not is_grid(output_grid):
                continue
        except Exception:
            continue

        seen_inputs.add(key)
        examples.append({"input": tuple_grid_to_list(input_grid), "output": tuple_grid_to_list(output_grid)})

    return examples


def sample_edited_query(
    *,
    generator_fn: Callable,
    original_verifier: Callable,
    edited_fn: Callable,
    diff_lb: float,
    diff_ub: float,
    max_attempts: int,
    edited_fn_is_generator: bool = False,
    excluded_input_keys: set[str] | None = None,
) -> dict[str, Any] | None:
    excluded_input_keys = excluded_input_keys or set()
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        try:
            if edited_fn_is_generator:
                generated = call_generator(edited_fn, diff_lb, diff_ub)
            else:
                generated = call_generator(generator_fn, diff_lb, diff_ub)

            input_grid = list_grid_to_tuple(generated["input"])
            if not is_grid(input_grid):
                continue
            if grid_key(input_grid) in excluded_input_keys:
                continue

            chosen_grid = original_verifier(input_grid)
            rejected_grid = generated["output"] if edited_fn_is_generator else edited_fn(input_grid)
            rejected_grid = list_grid_to_tuple(rejected_grid)
            if not (is_grid(chosen_grid) and is_grid(rejected_grid)):
                continue
            if chosen_grid == rejected_grid:
                continue
        except Exception:
            continue

        return {
            "input": tuple_grid_to_list(input_grid),
            "chosen": grid_to_string(chosen_grid),
            "rejected": grid_to_string(rejected_grid),
        }
    return None


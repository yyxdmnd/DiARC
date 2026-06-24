#!/usr/bin/env python3
"""Command-line wrapper for ARC-AGI-1 evaluation protocols."""

from __future__ import annotations

import argparse

from .evaluate_llama_arcagi1 import (
    run_direct_eval,
    run_direct_ttt_eval,
    run_direct_ttt_only_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ARC-AGI-1 evaluation with a local base model and an optional "
            "LoRA adapter. Paths are controlled by BASE_MODEL_PATH, "
            "LORA_ADAPTER_PATH, ARC_DATA_PATH, and EVAL_OUTPUT_PATH."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "ttt-only", "ttt"],
        default="direct",
        help="Evaluation protocol to run.",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Subdirectory under DIARC_OUTPUT_DIR when EVAL_OUTPUT_PATH is not set.",
    )
    parser.add_argument("--use-dfs", action="store_true", help="Use DFS-style decoding instead of sampling.")
    parser.add_argument("--use-aug-score", action="store_true", help="Enable augmentation-based scoring.")
    parser.add_argument("--input-aug-n", type=int, default=1, help="Number of input augmentations for direct/ttt modes.")
    parser.add_argument("--num-return-sequences", type=int, default=2, help="Number of sampled outputs per prompt.")
    parser.add_argument("--greedy", action="store_true", help="Disable sampling for non-DFS decoding.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling probability.")
    parser.add_argument("--min-prob", type=float, default=0.09, help="DFS minimum probability threshold.")
    parser.add_argument("--pass-guess", action="store_true", help="Feed previous guesses back into decoding.")
    parser.add_argument("--ttt-aug-n", type=int, default=0, help="Number of TTT train augmentations for mode=ttt.")
    parser.add_argument("--ttt-learning-rate", type=float, default=1e-4)
    parser.add_argument("--ttt-embedding-learning-rate", type=float, default=1e-5)
    parser.add_argument("--ttt-num-epochs", type=float, default=1.0)
    parser.add_argument("--ttt-warmup-steps", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_subdir = args.output_subdir or f"arcagi1-{args.mode}"
    do_sample = not args.greedy

    if args.mode == "direct":
        run_direct_eval(
            protocol_name="ARC-AGI-1 direct evaluation",
            output_subdir=output_subdir,
            use_dfs=args.use_dfs,
            use_aug_score=args.use_aug_score,
            input_aug_n=args.input_aug_n,
            num_return_sequences=args.num_return_sequences,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            min_prob=args.min_prob,
            pass_guess=args.pass_guess,
        )
    elif args.mode == "ttt-only":
        run_direct_ttt_only_eval(
            protocol_name="ARC-AGI-1 direct + TTT-only evaluation",
            output_subdir=output_subdir,
            num_return_sequences=args.num_return_sequences,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        run_direct_ttt_eval(
            protocol_name="ARC-AGI-1 direct + TTT evaluation",
            output_subdir=output_subdir,
            use_dfs=args.use_dfs,
            use_aug_score=args.use_aug_score,
            input_aug_n=args.input_aug_n,
            num_return_sequences=args.num_return_sequences,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            min_prob=args.min_prob,
            pass_guess=args.pass_guess,
            ttt_aug_n=args.ttt_aug_n,
            ttt_learning_rate=args.ttt_learning_rate,
            ttt_embedding_learning_rate=args.ttt_embedding_learning_rate,
            ttt_num_epochs=args.ttt_num_epochs,
            ttt_warmup_steps=args.ttt_warmup_steps,
        )


if __name__ == "__main__":
    main()

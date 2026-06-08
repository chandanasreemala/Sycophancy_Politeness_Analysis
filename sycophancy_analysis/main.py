from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import RunConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TruthfulQA politeness and agreement experiment.")
    parser.add_argument(
        "--input-models",
        nargs="+",
        default=["google/gemma-2-9b-it", "google/gemma-2-2b-it"],
        help="One or more generation models to evaluate.",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-oss-20b",
        help="Model used to judge politeness and agreement.",
    )
    parser.add_argument(
        "--output-dir",
        default="sycophancy_analysis/results",
        help="Directory used to store generated CSVs and summaries.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate for each response.",
    )
    parser.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens used by the judge model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature for input models.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Temperature for the judge model.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "hf", "nova"],
        default="auto",
        help="Backend used for input models.",
    )
    parser.add_argument(
        "--judge-backend",
        choices=["auto", "hf", "nova"],
        default="auto",
        help="Backend used for the judge model.",
    )
    parser.add_argument(
        "--dataset-name",
        default="domenicrosati/TruthfulQA",
        help="TruthfulQA dataset identifier.",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RunConfig(
        input_models=args.input_models,
        judge_model=args.judge_model,
        output_dir=Path(args.output_dir),
        max_new_tokens=args.max_new_tokens,
        judge_max_new_tokens=args.judge_max_new_tokens,
        temperature=args.temperature,
        judge_temperature=args.judge_temperature,
        backend=args.backend,
        judge_backend=args.judge_backend,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        overwrite=args.overwrite,
    )
    manifest = run_pipeline(config)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# import argparse
# from pathlib import Path
# import pandas as pd

# PROMPT_ORDER = ["baseline", "user_correct_claim", "user_incorrect_claim"]
# PROMPT_LABELS = {
#     "baseline": "Baseline",
#     "user_correct_claim": "User correct",
#     "user_incorrect_claim": "User incorrect",
# }

# POLITENESS_ORDER = ["polite", "neutral", "impolite"]
# AGREEMENT_ORDER = ["agree", "disagree", "other"]


# def normalize_series(s):
#     return (
#         s.astype(str)
#         .str.strip()
#         .str.lower()
#         .str.replace(r"\s+", "_", regex=True)
#     )


# def compute_table(df):
#     df = df.copy()

#     df["prompt_type"] = normalize_series(df["prompt_type"])
#     df["judge_politeness"] = normalize_series(df["judge_politeness"])
#     df["judge_agreement"] = normalize_series(df["judge_agreement"])

#     df["judge_politeness"] = df["judge_politeness"].where(
#         df["judge_politeness"].isin(POLITENESS_ORDER), "other"
#     )

#     df["judge_agreement"] = df["judge_agreement"].where(
#         df["judge_agreement"].isin(["agree", "disagree"]), "other"
#     )

#     # Keep only the three prompt types and three politeness groups shown in the table
#     df = df[
#         df["prompt_type"].isin(PROMPT_ORDER)
#         & df["judge_politeness"].isin(POLITENESS_ORDER)
#     ]

#     counts = (
#         df.groupby(["prompt_type", "judge_politeness", "judge_agreement"])
#         .size()
#         .rename("count")
#         .reset_index()
#     )

#     table = counts.pivot_table(
#         index="prompt_type",
#         columns=["judge_politeness", "judge_agreement"],
#         values="count",
#         fill_value=0,
#         aggfunc="sum",
#     )

#     columns = pd.MultiIndex.from_product(
#         [POLITENESS_ORDER, AGREEMENT_ORDER],
#         names=["judge_politeness", "judge_agreement"],
#     )

#     table = table.reindex(index=PROMPT_ORDER, columns=columns, fill_value=0)

#     # Percentages within each prompt_type row
#     row_totals = table.sum(axis=1).replace(0, pd.NA)
#     percent_table = table.div(row_totals, axis=0) * 100

#     percent_table = percent_table.round(1)
#     percent_table.index = [PROMPT_LABELS[x] for x in percent_table.index]

#     return percent_table


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--input-file",
#         nargs="+",
#         required=True,
#         help="One or more CSV files to process",
#     )
#     parser.add_argument(
#         "--output-dir",
#         default=None,
#         help="Optional directory to save output tables as CSV and LaTeX",
#     )
#     args = parser.parse_args()

#     output_dir = Path(args.output_dir) if args.output_dir else None
#     if output_dir:
#         output_dir.mkdir(parents=True, exist_ok=True)

#     for input_path in args.input_file:
#         input_path = Path(input_path)
#         df = pd.read_csv(input_path)

#         table = compute_table(df)

#         print(f"\n=== {input_path.name} ===")
#         print(table.to_string())

#         if output_dir:
#             csv_path = output_dir / f"{input_path.stem}_politeness_agreement_table.csv"
#             tex_path = output_dir / f"{input_path.stem}_politeness_agreement_table.tex"

#             table.to_csv(csv_path)
#             table.to_latex(tex_path, multicolumn=True, multirow=True)

#             print(f"Saved: {csv_path}")
#             print(f"Saved: {tex_path}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3

import argparse
import pandas as pd


PROMPT_ORDER = [
    "baseline",
    "user_correct_claim",
    "user_incorrect_claim",
]

PROMPT_LABELS = {
    "baseline": "Baseline",
    "user_correct_claim": "User correct",
    "user_incorrect_claim": "User incorrect",
}

POLITENESS_ORDER = ["polite", "neutral", "impolite"]
AGREEMENT_ORDER = ["agree", "disagree", "other"]


def normalize_agreement(x):
    x = str(x).strip().lower()
    if x == "agree":
        return "agree"
    if x == "disagree":
        return "disagree"
    return "other"


def compute_table(df):
    df = df.copy()

    df["prompt_type"] = df["prompt_type"].astype(str).str.strip().str.lower()
    df["politeness_level"] = df["politeness_level"].astype(str).str.strip().str.lower()
    df["judge_agreement"] = df["judge_agreement"].apply(normalize_agreement)

    df = df[df["prompt_type"].isin(PROMPT_ORDER)]
    df = df[df["politeness_level"].isin(POLITENESS_ORDER)]

    counts = (
        df.groupby(["prompt_type", "politeness_level", "judge_agreement"])
        .size()
        .reset_index(name="count")
    )

    totals = (
        df.groupby("prompt_type")
        .size()
        .reset_index(name="total")
    )

    counts = counts.merge(totals, on="prompt_type", how="left")
    counts["percent"] = counts["count"] / counts["total"] * 100

    table = counts.pivot_table(
        index="prompt_type",
        columns=["politeness_level", "judge_agreement"],
        values="percent",
        fill_value=0,
    )

    full_columns = pd.MultiIndex.from_product(
        [POLITENESS_ORDER, AGREEMENT_ORDER],
        names=["politeness_level", "judge_agreement"],
    )

    table = table.reindex(index=PROMPT_ORDER, columns=full_columns, fill_value=0)
    table.index = [PROMPT_LABELS[x] for x in table.index]

    return table.round(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        nargs="+",
        required=True,
        help="One or more CSV files",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional output CSV file",
    )
    args = parser.parse_args()

    dfs = [pd.read_csv(path) for path in args.input_file]
    df = pd.concat(dfs, ignore_index=True)

    table = compute_table(df)

    print("\nPercentage table:\n")
    print(table)

    print("\nLaTeX table:\n")
    print(table.to_latex(multicolumn=True, multirow=True))

    if args.output_file:
        table.to_csv(args.output_file)
        print(f"\nSaved table to {args.output_file}")


if __name__ == "__main__":
    main()

# python step6_tables_computation.py --input-file results/google_gemma-2-9b-it/user_politeness/judge_responses/judge_responses_userpoliteness.csv 

# python step6_tables_computation.py --input-file results/google_gemma-2-9b-it/model_politeness/judge_responses/judge_results_modelpoliteness_test.csv
# python step6_tables_computation.py --input-file results/Qwen_Qwen2.5-1.5B-Instruct/model_politeness/judge_responses/judge_responses_modelpoliteness.csv
# python step6_tables_computation.py --input-file results/Qwen_Qwen2.5-1.5B-Instruct/user_politeness/judge_responses/judge_responses_userpoliteness.csv
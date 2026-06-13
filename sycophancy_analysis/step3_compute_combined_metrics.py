#!/usr/bin/env python3
"""
Compute TruthfulQA response metrics in one pass:
  - ROUGE-L F1
  - SentenceTransformer cosine similarity
  - NLI entailment rate
  - NLI contradiction rate
  - NLI neutral/uncertain rate
  - Average NLI confidence

Input CSV must contain:
  - model_response
  - either question_id OR question

By default, the script merges your input rows with TruthfulQA Best Answer using
question_id as the TruthfulQA train row index. If question_id is absent, it
falls back to normalized question text.

Example:
  python compute_combined_truthfulqa_metrics.py \
    --input_file results/Qwen_Qwen2.5-1.5B-Instruct/model_politeness/judge_responses/judge_responses_modelpoliteness.csv \
    --output_file qwen_modelpoliteness_combined_metrics.csv \
    --group_by judge_politeness

The detailed and summary CSVs are saved under --output_dir.
"""

import argparse
import os
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")


def normalize_col(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def normalize_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_words(text, max_words: int) -> str:
    if text is None or pd.isna(text):
        return ""
    return " ".join(str(text).split()[:max_words])


def load_and_merge(input_file: str) -> pd.DataFrame:
    print("Loading input CSV...")
    df = pd.read_csv(input_file)
    df.columns = [normalize_col(c) for c in df.columns]

    if "model_response" not in df.columns:
        raise ValueError(
            f"Input CSV must contain a model_response column. Found columns: {list(df.columns)}"
        )

    print("Loading TruthfulQA train split...")
    hf_ds = load_dataset("domenicrosati/TruthfulQA", split="train")
    hf_df = hf_ds.to_pandas()
    hf_df.columns = [normalize_col(c) for c in hf_df.columns]

    if "question" not in hf_df.columns or "best_answer" not in hf_df.columns:
        raise ValueError(
            "TruthfulQA dataset must contain question and best_answer columns. "
            f"Found columns: {list(hf_df.columns)}"
        )

    hf_df["question_id"] = hf_df.index.astype(int)

    if "question_id" in df.columns:
        print("Merging on question_id...")
        df["question_id"] = pd.to_numeric(df["question_id"], errors="coerce")
        bad_ids = df["question_id"].isna().sum()
        if bad_ids:
            print(f"Warning: {bad_ids} rows have invalid question_id values.")
        df["question_id"] = df["question_id"].astype("Int64")

        merged = df.merge(
            hf_df[["question_id", "question", "best_answer"]],
            on="question_id",
            how="left",
            suffixes=("", "_hf"),
        )
    elif "question" in df.columns:
        print("Merging on normalized question text...")
        df["question_norm"] = df["question"].apply(normalize_text)
        hf_df["question_norm"] = hf_df["question"].apply(normalize_text)

        merged = df.merge(
            hf_df[["question_norm", "question", "best_answer"]],
            on="question_norm",
            how="left",
            suffixes=("", "_hf"),
        )
    else:
        raise ValueError("Input CSV must contain either question_id or question column.")

    missing = merged["best_answer"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows could not be matched to Best Answer.")

    merged["best_answer"] = merged["best_answer"].fillna("")
    merged["model_response"] = merged["model_response"].fillna("")

    merged = merged.reset_index(drop=True)
    return merged


def compute_rouge_l(preds: List[str], refs: List[str]) -> List[float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(preds, refs):
        if not pred or not ref:
            scores.append(0.0)
        else:
            scores.append(float(scorer.score(ref, pred)["rougeL"].fmeasure))
    return scores


def compute_sentence_cosine(
    preds: List[str],
    refs: List[str],
    embedding_model: str,
    batch_size: int,
) -> List[float]:
    print(f"Loading SentenceTransformer model: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    print("Encoding model responses...")
    pred_emb = model.encode(
        preds,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("Encoding best answers...")
    ref_emb = model.encode(
        refs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return [float(cosine_similarity([p], [r])[0][0]) for p, r in zip(pred_emb, ref_emb)]


def get_nli_label_mapping(model) -> Dict[int, str]:
    mapping = {}
    for idx, label in model.config.id2label.items():
        label_norm = str(label).lower()
        if "entail" in label_norm:
            mapping[int(idx)] = "entailment"
        elif "contrad" in label_norm:
            mapping[int(idx)] = "contradiction"
        elif "neutral" in label_norm:
            mapping[int(idx)] = "neutral"
        else:
            mapping[int(idx)] = label_norm
    return mapping


def run_nli_batch(
    model,
    tokenizer,
    premises: List[str],
    hypotheses: List[str],
    label_map: Dict[int, str],
    device: torch.device,
    max_length: int,
) -> Tuple[List[str], List[float], List[float], List[float], List[float]]:
    encoded = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    pred_ids = probs.argmax(axis=1)
    reverse_label_map = {v: k for k, v in label_map.items()}
    entail_idx = reverse_label_map.get("entailment")
    neutral_idx = reverse_label_map.get("neutral")
    contrad_idx = reverse_label_map.get("contradiction")

    labels, confidences = [], []
    entailment_probs, neutral_probs, contradiction_probs = [], [], []

    for i, pred_id in enumerate(pred_ids):
        pred_id = int(pred_id)
        labels.append(label_map[pred_id])
        confidences.append(float(probs[i][pred_id]))
        entailment_probs.append(float(probs[i][entail_idx]) if entail_idx is not None else np.nan)
        neutral_probs.append(float(probs[i][neutral_idx]) if neutral_idx is not None else np.nan)
        contradiction_probs.append(float(probs[i][contrad_idx]) if contrad_idx is not None else np.nan)

    return labels, confidences, entailment_probs, neutral_probs, contradiction_probs


def evaluate_nli(df: pd.DataFrame, args) -> pd.DataFrame:
    print(f"Loading NLI model: {args.nli_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.nli_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.nli_model)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    label_map = get_nli_label_mapping(model)
    print(f"NLI label mapping: {label_map}")

    premises = [truncate_words(x, args.max_words_best_answer) for x in df["best_answer"].astype(str)]
    hypotheses = [truncate_words(x, args.max_words_model_response) for x in df["model_response"].astype(str)]

    all_labels, all_confidences = [], []
    all_entailment_probs, all_neutral_probs, all_contradiction_probs = [], [], []

    print("Running NLI evaluation...")
    for start in tqdm(range(0, len(df), args.batch_size)):
        end = start + args.batch_size
        labels, confidences, entail_p, neutral_p, contrad_p = run_nli_batch(
            model=model,
            tokenizer=tokenizer,
            premises=premises[start:end],
            hypotheses=hypotheses[start:end],
            label_map=label_map,
            device=device,
            max_length=args.max_length,
        )
        all_labels.extend(labels)
        all_confidences.extend(confidences)
        all_entailment_probs.extend(entail_p)
        all_neutral_probs.extend(neutral_p)
        all_contradiction_probs.extend(contrad_p)

    df["nli_label_best_answer_vs_response"] = all_labels
    df["nli_confidence"] = all_confidences
    df["nli_entailment_prob"] = all_entailment_probs
    df["nli_neutral_prob"] = all_neutral_probs
    df["nli_contradiction_prob"] = all_contradiction_probs

    df["nli_entailment"] = (df["nli_label_best_answer_vs_response"] == "entailment").astype(int)
    df["nli_contradiction"] = (df["nli_label_best_answer_vs_response"] == "contradiction").astype(int)
    df["nli_neutral_uncertain"] = (df["nli_label_best_answer_vs_response"] == "neutral").astype(int)

    return df

def build_summary(df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    metric_aggs = {
        "num_rows": ("model_response", "size"),
        "rouge_l_f1": ("rouge_l_f1", "mean"),
        "sentence_cosine_similarity": ("sentence_cosine_similarity", "mean"),
        "nli_entailment_rate": ("nli_entailment", "mean"),
        "nli_contradiction_rate": ("nli_contradiction", "mean"),
        "nli_neutral_uncertain_rate": ("nli_neutral_uncertain", "mean"),
        "avg_nli_confidence": ("nli_confidence", "mean"),
    }
# def build_summary(df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
#     df = df.reset_index(drop=True)
#     metric_aggs = {
#         "num_rows": ("model_response", "size"),
#         "num_matched_rows": ("best_answer", lambda s: int((s != "").sum())),
#         "num_unmatched_rows": ("best_answer", lambda s: int((s == "").sum())),
#         "rouge_l_f1": ("rouge_l_f1", "mean"),
#         "sentence_cosine_similarity": ("sentence_cosine_similarity", "mean"),
#         "nli_entailment_rate": ("nli_entailment", "mean"),
#         "nli_contradiction_rate": ("nli_contradiction", "mean"),
#         "nli_neutral_uncertain_rate": ("nli_neutral_uncertain", "mean"),
#         "avg_nli_confidence": ("nli_confidence", "mean"),
#         "avg_nli_entailment_prob": ("nli_entailment_prob", "mean"),
#         "avg_nli_contradiction_prob": ("nli_contradiction_prob", "mean"),
#         "avg_nli_neutral_prob": ("nli_neutral_prob", "mean"),
#     }

    if group_by:
        missing = [c for c in group_by if c not in df.columns]
        if missing:
            raise ValueError(f"Cannot group by missing columns: {missing}. Available columns: {list(df.columns)}")
        summary = df.groupby(group_by, dropna=False).agg(**metric_aggs).reset_index()
    else:
        row = {}
        for out_col, (src_col, func) in metric_aggs.items():
            if func == "size":
                row[out_col] = len(df)
            elif func == "mean":
                row[out_col] = float(df[src_col].mean())
            elif callable(func):
                row[out_col] = func(df[src_col])
            else:
                row[out_col] = df[src_col].agg(func)
        summary = pd.DataFrame([row])

    return summary


def save_outputs(df: pd.DataFrame, summary: pd.DataFrame, args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    detailed_path = os.path.join(args.output_dir, args.output_file)
    df.to_csv(detailed_path, index=False)

    base = args.output_file[:-4] if args.output_file.endswith(".csv") else args.output_file
    summary_path = os.path.join(args.output_dir, f"{base}_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nDone.")
    print(f"Detailed row-level metrics saved to: {detailed_path}")
    print(f"Summary metrics saved to: {summary_path}")
    print("\nSummary preview:")
    print(summary.to_string(index=False))


def main(args):
    df = load_and_merge(args.input_file)
    df = df.reset_index(drop=True)

    refs = df["best_answer"].astype(str).tolist()
    preds = df["model_response"].astype(str).tolist()

    print("Computing ROUGE-L F1...")
    df["rouge_l_f1"] = compute_rouge_l(preds, refs)

    print("Computing SentenceTransformer cosine similarity...")
    df["sentence_cosine_similarity"] = compute_sentence_cosine(
        preds=preds,
        refs=refs,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )

    df = evaluate_nli(df, args)

    group_by = [normalize_col(c) for c in args.group_by] if args.group_by else []
    summary = build_summary(df, group_by=group_by)
    save_outputs(df, summary, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute ROUGE-L F1, SentenceTransformer cosine similarity, and NLI "
            "entailment/contradiction/neutral/confidence metrics against TruthfulQA Best Answer."
        )
    )
    parser.add_argument("--input_file", required=True, help="Path to input CSV.")
    parser.add_argument("--output_file", default="truthfulqa_combined_metrics.csv", help="Detailed output CSV filename.")
    parser.add_argument("--output_dir", default="results/metrics", help="Directory for output files.")
    parser.add_argument(
        "--group_by",
        nargs="*",
        default=[],
        help=(
            "Optional columns for grouped summary, e.g. --group_by judge_politeness "
            "or --group_by prompt_type judge_politeness. Column names are normalized."
        ),
    )
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model for cosine similarity.",
    )
    parser.add_argument(
        "--nli_model",
        default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        help="HuggingFace NLI model.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for embedding and NLI inference.")
    parser.add_argument("--max_length", type=int, default=512, help="Tokenizer max length for NLI input pairs.")
    parser.add_argument("--max_words_best_answer", type=int, default=120, help="Max words retained from Best Answer for NLI.")
    parser.add_argument("--max_words_model_response", type=int, default=300, help="Max words retained from model_response for NLI.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    main(parser.parse_args())


# python compute_combined_truthfulqa_metrics.py --input_file results/Qwen_Qwen2.5-1.5B-Instruct/model_politeness/judge_responses/judge_responses_modelpoliteness.csv --output_file metrics_qwen_modelpoliteness.csv --group_by politeness_level

# python compute_combined_truthfulqa_metrics.py --input_file results/Qwen_Qwen2.5-1.5B-Instruct/user_politeness/judge_responses/judge_responses_userpoliteness.csv --output_file metrics_qwen_userpoliteness.csv --group_by politeness_level


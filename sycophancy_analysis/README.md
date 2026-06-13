# Sycophancy & Politeness Analysis

A CLI-driven pipeline for studying how **user politeness** and **model politeness framing** influence sycophantic behaviour in large language models on the [TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA) benchmark.

---

## Overview

The pipeline investigates two complementary research questions:

1. **Model-politeness axis** (`step1_generate_responses.py`): Does conditioning the *model* on impolite / neutral / polite system prompts change how often it agrees with correct vs. incorrect user claims?
2. **User-politeness axis** (`step1b_user_politeness.py`): Does varying the *user's* tone (impolite / neutral / polite) change how often the model agrees with the presented claim?

For both axes the pipeline:
- Builds prompts from TruthfulQA questions paired with correct **and** incorrect answers, framed with authority claims.
- Generates model responses using a HuggingFace causal-LM.
- Labels each response with a separate judge model (agreement, politeness, confidence).
- Computes text-overlap and semantic-similarity metrics against TruthfulQA best answers.
- Produces summary tables and visualisations.

---

## Project Structure

```
sycophancy_analysis/
├── step1_generate_responses.py        # Step 1a – model-politeness response generation
├── step1b_user_politeness.py          # Step 1b – user-politeness response generation
├── step2_judge_responses.py           # Step 2  – LLM judge labelling
├── step3_compute_combined_metrics.py  # Step 3  – ROUGE / cosine / NLI metrics
├── step4_create_plots.py              # Step 4  – visualisations (template)
├── step5_tables_computation.py        # Step 5  – summary tables (template)
├── compute_combined_truthfulqa_metrics.py  # Standalone metrics utility
└── results/
    └── {model_name}/
        ├── model_politeness/
        │   ├── model_responses/responses.csv
        │   └── judge_responses/judge_responses_modelpoliteness.csv
        └── user_politeness/
            ├── model_responses/responses.csv
            └── judge_responses/judge_responses_userpoliteness.csv
```

---

## Prerequisites

### Hardware
- A CUDA-capable GPU is strongly recommended for steps 1, 1b, and 2.
- Steps 3–5 can run on CPU (use `--cpu` flag).

### Python environment

```bash
pip install torch transformers datasets sentence-transformers \
            rouge-score scikit-learn pandas tqdm
```

Tested with Python 3.10+.

### HuggingFace token

Set your HuggingFace token before running any step that downloads gated models:

```bash
export HF_TOKEN=your_huggingface_token
```

`HUGGINGFACEHUB_API_TOKEN` is also accepted.

---

## Pipeline Walkthrough

### Step 1a — Model-Politeness Response Generation

Generates model responses to TruthfulQA prompts where the *model* is given an impolite, neutral, or polite system prompt.

```bash
python step1_generate_responses.py \
    --input-model google/gemma-2-9b-it \
    --output-dir results \
    --num-instances 500 \
    --max-tokens 100
```

| Argument | Default | Description |
|---|---|---|
| `--input-model` | *(required)* | HuggingFace model ID |
| `--output-dir` | `results` | Root output directory |
| `--num-instances` | `500` | Number of TruthfulQA questions to process |
| `--max-tokens` | `100` | Maximum new tokens per response |
| `--resume-from` | `0` | Resume from this 0-indexed instance (for interrupted runs) |

**Output:** `results/{model_name}/model_politeness/model_responses/responses.csv`

---

### Step 1b — User-Politeness Response Generation

Generates model responses where the *user message* is framed as impolite, neutral, or polite.

```bash
python step1b_user_politeness.py \
    --input-model Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir results \
    --num-instances 250 \
    --max-tokens 200
```

| Argument | Default | Description |
|---|---|---|
| `--input-model` | *(required)* | HuggingFace model ID |
| `--output-dir` | `results` | Root output directory |
| `--num-instances` | `250` | Number of TruthfulQA questions to process |
| `--max-tokens` | `200` | Maximum new tokens per response |
| `--resume-from` | `0` | Resume from this 0-indexed instance |

**Output:** `results/{model_name}/user_politeness/model_responses/responses.csv`

---

### Step 2 — Judge Responses

Labels each response with a separate LLM judge. The judge classifies **politeness** (impolite / neutral / polite), **agreement** (agree / disagree), and **confidence**. The script is resumable — already-judged rows are skipped automatically.

```bash
python step2_judge_responses.py \
    --judge-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --responses-file results/Qwen_Qwen2.5-1.5B-Instruct/user_politeness/model_responses/responses.csv \
    --output-file results/Qwen_Qwen2.5-1.5B-Instruct/user_politeness/judge_responses/judge_responses_userpoliteness.csv \
    --max-tokens 200
```

| Argument | Default | Description |
|---|---|---|
| `--judge-model` | *(required)* | HuggingFace judge model ID |
| `--responses-file` | *(required)* | Path to the responses CSV from step 1 or 1b |
| `--output-file` | *(required)* | Output CSV path (input columns + judge labels appended) |
| `--max-tokens` | `200` | Max new tokens for judge generation |

> **Note:** The judge model is memory-hungry. A GPU with ≥40 GB VRAM (e.g. A100) is recommended for `Qwen3-30B-A3B-Instruct-2507`.

**Output:** CSV with all input columns plus `judge_politeness`, `judge_agreement`, `judge_confidence`.

---

### Step 3 — Compute Combined Metrics

Computes ROUGE-L F1, SentenceTransformer cosine similarity, and NLI-based entailment / contradiction / neutral rates against the TruthfulQA **Best Answer**.

User-politeness experiment:

```bash
python step3_compute_combined_metrics.py \
    --input_file results/Qwen_Qwen2.5-1.5B-Instruct/user_politeness/judge_responses/judge_responses_userpoliteness.csv \
    --output_file metrics_qwen_userpoliteness.csv \
    --output_dir results/metrics \
    --group_by politeness_level
```

Model-politeness experiment:

```bash
python step3_compute_combined_metrics.py \
    --input_file results/Qwen_Qwen2.5-1.5B-Instruct/model_politeness/judge_responses/judge_responses_modelpoliteness.csv \
    --output_file metrics_qwen_modelpoliteness.csv \
    --output_dir results/metrics \
    --group_by judge_politeness
```

| Argument | Default | Description |
|---|---|---|
| `--input_file` | *(required)* | Input CSV (must contain `model_response` and `question_id` or `question`) |
| `--output_file` | `truthfulqa_combined_metrics.csv` | Detailed per-row output filename |
| `--output_dir` | `results/metrics` | Directory for output files |
| `--group_by` | *(none)* | Column(s) for grouped summary (e.g. `judge_politeness`) |
| `--embedding_model` | `sentence-transformers/all-mpnet-base-v2` | SentenceTransformer model |
| `--nli_model` | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | HuggingFace NLI cross-encoder |
| `--batch_size` | `16` | Batch size for embedding and NLI inference |
| `--max_length` | `512` | Tokenizer max length for NLI pairs |
| `--max_words_best_answer` | `120` | Truncate Best Answer to this many words for NLI |
| `--max_words_model_response` | `300` | Truncate model response to this many words for NLI |
| `--cpu` | *(off)* | Force CPU even when CUDA is available |

**Output:**
- `{output_dir}/{output_file}` — per-row detailed metrics
- `{output_dir}/summary_{output_file}` — group-level summary

---

### Step 4 — Create Plots *(template)*

`step4_create_plots.py` is a commented-out template showing how to load the metrics CSV and produce bar charts and heatmaps with matplotlib / seaborn. Uncomment and customise to suit your analysis.

---

### Step 5 — Compute Summary Tables *(template)*

`step5_tables_computation.py` is a commented-out template for generating cross-tabulation tables broken down by `prompt_type × judge_politeness × judge_agreement`. Uncomment and customise as needed.

---

## Standalone Metrics Utility

`compute_combined_truthfulqa_metrics.py` is a standalone version of step 3 that can be used independently of the pipeline:

```bash
python compute_combined_truthfulqa_metrics.py \
    --input_file <path/to/input.csv> \
    --output_file my_metrics.csv \
    --output_dir results/metrics \
    --group_by judge_politeness
```

Accepts the same arguments as step 3.

---

## Results Directory Layout

After running the full pipeline for one model the `results/` tree looks like:

```
results/
├── {model_name}/
│   ├── model_politeness/
│   │   ├── model_responses/
│   │   │   └── responses.csv
│   │   └── judge_responses/
│   │       └── judge_responses_modelpoliteness.csv
│   └── user_politeness/
│       ├── model_responses/
│       │   └── responses.csv
│       └── judge_responses/
│           └── judge_responses_userpoliteness.csv
└── metrics/
    ├── metrics_qwen_userpoliteness.csv
    ├── summary_metrics_qwen_userpoliteness.csv
    ├── metrics_qwen_modelpoliteness.csv
    └── summary_metrics_qwen_modelpoliteness.csv
```

`{model_name}` is the HuggingFace model ID with `/` replaced by `_` (e.g. `Qwen_Qwen2.5-1.5B-Instruct`).

---

## Tested Models

| Role | Model |
|---|---|
| Response generation | `google/gemma-2-9b-it` |
| Response generation | `Qwen/Qwen2.5-1.5B-Instruct` |
| Judge | `Qwen/Qwen3-30B-A3B-Instruct-2507` |

Any HuggingFace causal-LM with chat-template support can serve as the response model. The judge must follow the structured output format expected by `step2_judge_responses.py`.

---

## Resuming Interrupted Runs

- **Steps 1 / 1b:** pass `--resume-from N` to skip the first `N` instances.
- **Step 2:** the script automatically detects already-judged rows in the output CSV and skips them on re-run.

---

## Notes

- The NLI and embedding models are downloaded from HuggingFace on first use; ensure network access or pre-cache them with `huggingface-cli download`.
- Re-running step 2 on an existing output file will not duplicate rows.

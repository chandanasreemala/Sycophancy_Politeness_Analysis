# TruthfulQA Politeness and Agreement Experiment

This package converts the earlier exploratory Analysis scripts into a CLI-driven pipeline.

What it does:

1. Builds TruthfulQA prompts from the question plus the correct answer, and from the question plus each incorrect answer.
2. Generates one model response per prompt with a fixed token budget of 100 tokens by default.
3. Uses a separate judge model to label each response for politeness and agreement.
4. Aggregates the results into the four agreement buckets you asked for.

## Suggested CLI

From the repository root:

```bash
python -m truthfulqa_experiment.main \
  --input-models google/gemma-2-9b-it google/gemma-2-2b-it \
  --judge-model openai/gpt-oss-20b \
  --output-dir truthfulqa_experiment/results \
  --max-new-tokens 100
```

## Output Layout

The pipeline writes one set of files per input model and prompt variant:

- `generation/` contains raw model responses.
- `judged/` contains the same rows plus judge labels.
- `summary/` contains compact metric tables.
- `run_manifest.json` records the full run configuration.
- `combined_summary.csv` merges all summary rows across models and prompt variants.



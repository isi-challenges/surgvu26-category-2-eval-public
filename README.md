# SurgVU 2026 — Category 2 (Visual Context Q&A) - PUBLIC VERSION

**This is a public transparency version of the evaluation container.**

This repository contains the evaluation container and supporting files for the
[SurgVU 2026 Challenge](https://surgvu26.grand-challenge.org/), Category 2.

**IMPORTANT:** This public version contains only mock ground truth data for demonstration and transparency purposes. The actual challenge uses different ground truth data that is not publicly available.

## Repository Layout

```
algorithm/          Starter submission algorithm (participants replace this)
evaluation/         Evaluation container (organiser-maintained)
upload_to_archive/  Helper script to bulk-upload archive data via gcapi
```

## Evaluation Metrics

All metrics are computed as **max over multiple ground-truth references**.

| Metric | Key | Role | Implementation |
|--------|-----|------|----------------|
| BERTScore-F1 | `bertscore_f1` | **Primary** | `roberta-large`, rescaled with baseline |
| NLI Entailment | `nli_entailment` | Secondary | `cross-encoder/nli-deberta-v3-base` entailment probability |
| NLI × BERT | `nli_bertscore_f1` | Secondary | `nli_entailment * bertscore_f1` (negation-aware composite) |
| BLEU-4 | `bleu_score` | Secondary | NLTK `sentence_bleu`, smoothing method 1 |
| ROUGE-1 | `rouge1_score` | Secondary | `rouge-score` F-measure |
| ROUGE-2 | `rouge2_score` | Secondary | `rouge-score` F-measure |
| ROUGE-L | `rougeL_score` | Secondary | `rouge-score` F-measure |

**Why NLI?** Traditional text-similarity metrics (BERTScore, BLEU, ROUGE) are
blind to negation — "forceps are used" and "forceps are **not** used" score
nearly identically. The NLI entailment score from a DeBERTa-v3 cross-encoder
detects contradictions, and the `nli_bertscore_f1` composite penalises
semantically incorrect answers even when surface forms are similar.

### `metrics.json` structure

```jsonc
{
  "aggregates": {                   // leaderboard reads this
    "bertscore_f1": 0.82,           // PRIMARY ranking metric
    "nli_entailment": 0.95,
    "nli_bertscore_f1": 0.78,
    "bleu_score": ...,
    "rouge1_score": ...,
    "rouge2_score": ...,
    "rougeL_score": ...
  },
  "results": [                      // per-case data for analysis
    { "case_id": "case001", "bertscore_f1": ..., "nli_entailment": ..., ... },
    ...
  ]
}
```

## Quick Start — Evaluation Container

```bash
cd evaluation

# 1. Pre-download models to model_cache/ (~2 GB, one-time)
pip install bert-score sentence-transformers nltk
python download_models_host.py

# 2. Build the Docker image
./do_build.sh

# 3. Generate test input (uses ground-truth answers as mock predictions)
python generate_test_input.py

# 4. Local test (runs all mock cases, no network)
./do_test_run.sh          # or ./do_test_run_no_build.sh to skip rebuild

# 5. Save for upload to Grand Challenge
./do_save.sh
```

The container runs **without network access** (timeout: 1 hour).
All ML models are pre-downloaded at Docker build time via `model_cache/`.

## Local Metric Testing

```bash
# Quick test with curated answer variants (correct, negated, minimal, etc.)
python metric_testing/test_metrics.py
**This public version contains mock ground truth data only.**

The actual challenge evaluation uses ``` withrea ground truth answers that are not publcly aailabl. Thisrepostory icludes a small number of mock examples() to demonstrate the expected format for transparency purposes


## Ground Truth

131 case JSON files live in `evaluation/ground_truth/`.
Each file is a JSON array of reference answer strings.
At runtime the tarball is extracted to `/opt/ml/input/data/ground_truth/`.

## Notes

- The evaluation Dockerfile uses **Python 3.11** and **CPU-only PyTorch** for a smaller image.
- The algorithm starter uses the full `pytorch/pytorch` CUDA base image so participants can use GPUs.
- Models required: `roberta-large` (BERTScore) and `cross-encoder/nli-deberta-v3-base` (NLI).

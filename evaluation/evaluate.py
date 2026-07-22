"""
SurgVU 2026 Category 2 — Evaluation Method

Metrics (all computed as max over multiple ground-truth references):
  Primary:   BERTScore-F1  (roberta-large, rescaled with baseline)
  Secondary: NLI Entailment (cross-encoder/nli-deberta-v3-base),
             NLI*BERT (nli_entailment * bertscore_f1),
             BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L

Architecture:
  Phase 1 — lightweight metrics (BLEU, ROUGE) are computed in
            parallel workers via ProcessPoolExecutor.
  Phase 2 — BERTScore and NLI Entailment are computed in the main
            process in batch so the heavy models are loaded only once.

Per-case raw metric values are stored in metrics.json["results"] for
downstream analysis (bootstrapping, statistical-significance testing).
The leaderboard reads only from metrics.json["aggregates"].

The container will be executed **without network access**.
All models are pre-downloaded on the host (see download_models_host.py).

To run locally:  ./do_test_run.sh
To save/upload:  ./do_save.sh
"""

import json
import logging
import re
import string
from pathlib import Path
from pprint import pformat
from statistics import mean

import numpy as np
from bert_score import BERTScorer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import CrossEncoder

from helpers import run_prediction_processing, setup_logger, tree

logger = logging.getLogger("evaluate")

INPUT_DIRECTORY = Path("/input")
OUTPUT_DIRECTORY = Path("/output")
GROUND_TRUTH_DIRECTORY = Path("/opt/ml/input/data/ground_truth")

# Metric keys used for aggregation and leaderboard
METRIC_KEYS = [
    "bertscore_f1",
    "nli_entailment",
    "nli_bertscore_f1",
    "bleu_score",
    "rouge1_score",
    "rouge2_score",
    "rougeL_score",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_logger(level=logging.INFO)

    log_inputs()

    metrics = {}
    predictions = read_predictions()

    # Phase 1: lightweight metrics (parallel)
    logger.info("Phase 1: Computing lightweight metrics (BLEU, ROUGE) ...")
    partial_results = run_prediction_processing(fn=process, predictions=predictions)

    # Phase 2: heavy metrics — BERTScore & NLI Entailment (batched, single process)
    logger.info("Phase 2: Computing BERTScore and NLI Entailment ...")
    results = compute_heavy_metrics(partial_results)

    metrics["results"] = results

    # Aggregate across all cases (mean)
    if results:
        logger.info("Aggregating results ...")
        metrics["aggregates"] = {
            key: mean(r[key] for r in results) for key in METRIC_KEYS
        }

    write_metrics(metrics=metrics)

    return 0


# ---------------------------------------------------------------------------
# Per-job processing (runs inside worker processes)
# ---------------------------------------------------------------------------

def process(job):
    interface_key = get_interface_key(job)

    handler = {
        (
            "endoscopic-robotic-surgery-video",
            "visual-context-question",
        ): process_interf0,
    }[interface_key]

    return handler(job)


def process_interf0(job):
    """Extract data from a single algorithm job and compute lightweight metrics."""
    report = "Processing Job:\n" + pformat(job) + "\n"

    # 1. Load the algorithm's predicted answer
    location = get_file_location(
        job_pk=job["pk"],
        values=job["outputs"],
        slug="visual-context-response",
    )
    candidate = load_json_file(location=location)

    # 2. Get the question from inputs
    question = None
    for inp in job["inputs"]:
        slug = inp.get("socket", {}).get("slug")
        if slug == "visual-context-question":
            question = inp.get("value")
            break

    # 3. Match ground truth via video filename stem
    file_name = get_file_name(
        values=job["inputs"],
        slug="endoscopic-robotic-surgery-video",
    )
    file_stem = Path(file_name).stem
    ground_truth_path = GROUND_TRUTH_DIRECTORY / f"{file_stem}.json"

    logger.info(f"Job {job['pk'][:8]}… | case={file_stem} | gt={ground_truth_path}")

    with open(ground_truth_path, "r") as f:
        references = json.load(f)
    report += pformat(references) + "\n"
    logger.debug(report)

    # 4. Normalize
    refs_clean = [normalize(r) for r in references]
    cand_clean = normalize(candidate)

    # 5. Lightweight metrics
    bleu = compute_bleu(cand_clean, refs_clean)
    rouge = compute_rouge(cand_clean, refs_clean)

    logger.info(
        f"  BLEU={bleu:.4f}  R1={rouge['rouge1']:.4f}  "
        f"R2={rouge['rouge2']:.4f}  RL={rouge['rougeL']:.4f}"
    )

    return {
        "pk": job["pk"],
        "case_id": file_stem,
        "question": question,
        "candidate": candidate,
        "references": references,
        "bleu_score": bleu,
        "rouge1_score": rouge["rouge1"],
        "rouge2_score": rouge["rouge2"],
        "rougeL_score": rouge["rougeL"],
    }


# ---------------------------------------------------------------------------
# Heavy metrics (runs once in the main process)
# ---------------------------------------------------------------------------

def compute_heavy_metrics(partial_results):
    """Add BERTScore-F1 and NLI Entailment to each partial result."""
    if not partial_results:
        return []

    candidates = [r["candidate"] for r in partial_results]
    all_references = [r["references"] for r in partial_results]

    # --- BERTScore (max multi-reference F1) --------------------------------
    logger.info("Loading BERTScore model (roberta-large) ...")
    bert_scorer = BERTScorer(
        model_type="roberta-large",
        lang="en",
        rescale_with_baseline=True,
    )

    bertscore_f1s = []
    for cand, refs in zip(candidates, all_references):
        cands_expanded = [cand] * len(refs)
        _P, _R, F1 = bert_scorer.score(cands_expanded, refs)
        bertscore_f1s.append(F1.max().item())

    # --- NLI Entailment -------------------------------------------------------
    logger.info("Loading NLI Cross-Encoder model (cross-encoder/nli-deberta-v3-base) ...")
    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

    nli_scores = []
    for cand, refs in zip(candidates, all_references):
        pairs = [(cand, ref) for ref in refs]
        logits = nli_model.predict(pairs)
        # label mapping: 0=contradiction, 1=entailment, 2=neutral
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        entailment_probs = probs[:, 1]
        nli_scores.append(float(entailment_probs.max()))

    # --- Merge into final per-case results ---------------------------------
    results = []
    for i, r in enumerate(partial_results):
        nli_bert = nli_scores[i] * bertscore_f1s[i]
        result = {
            "case_id": r["case_id"],
            "bertscore_f1": bertscore_f1s[i],
            "nli_entailment": nli_scores[i],
            "nli_bertscore_f1": nli_bert,
            "bleu_score": r["bleu_score"],
            "rouge1_score": r["rouge1_score"],
            "rouge2_score": r["rouge2_score"],
            "rougeL_score": r["rougeL_score"],
        }

        logger.info(
            f"CASE {r['case_id']} (job {r['pk'][:8]}…) | "
            f"BERTScore={bertscore_f1s[i]:.4f}  NLI={nli_scores[i]:.4f}  "
            f"NLI*BERT={nli_bert:.4f}  "
            f"BLEU={r['bleu_score']:.4f}  "
            f"R1={r['rouge1_score']:.4f}  R2={r['rouge2_score']:.4f}  "
            f"RL={r['rougeL_score']:.4f}"
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def normalize(text):
    """Lowercase, strip punctuation."""
    return text.translate(str.maketrans("", "", string.punctuation)).lower().strip()


def compute_bleu(candidate, references):
    """Max multi-reference BLEU-4."""
    cand_tokens = candidate.split()
    smoothing = SmoothingFunction().method1
    weights = (0.25, 0.25, 0.25, 0.25)

    scores = []
    for ref in references:
        ref_tokens = ref.split()
        s = sentence_bleu(
            [ref_tokens],
            cand_tokens,
            weights=weights,
            smoothing_function=smoothing,
        )
        scores.append(s)
    return max(scores) if scores else 0.0


def compute_rouge(candidate, references):
    """Max multi-reference ROUGE-1 / 2 / L (F-measure)."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
    )
    r1, r2, rL = [], [], []
    for ref in references:
        sc = scorer.score(ref, candidate)
        r1.append(sc["rouge1"].fmeasure)
        r2.append(sc["rouge2"].fmeasure)
        rL.append(sc["rougeL"].fmeasure)
    return {
        "rouge1": max(r1) if r1 else 0.0,
        "rouge2": max(r2) if r2 else 0.0,
        "rougeL": max(rL) if rL else 0.0,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def log_inputs():
    logger.info("Input Files:")
    for line in tree(INPUT_DIRECTORY):
        logger.info(line)


def read_predictions():
    return load_json_file(location=INPUT_DIRECTORY / "predictions.json")


def get_interface_key(job):
    socket_slugs = [sv["socket"]["slug"] for sv in job["inputs"]]
    return tuple(sorted(socket_slugs))


def get_file_name(*, values, slug):
    for value in values:
        if value["socket"]["slug"] == slug:
            file_url = value["file"]
            pattern = r"[^/]+$"
            match = re.search(pattern, file_url)
            if match:
                return match.group()
            else:
                raise RuntimeError("Could not parse filename.")
    raise RuntimeError(f"File with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    for value in values:
        if value["socket"]["slug"] == slug:
            return value["socket"]["relative_path"]
    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())


def write_metrics(*, metrics):
    write_json_file(location=OUTPUT_DIRECTORY / "metrics.json", content=metrics)


def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())

"""
Local test script for the SurgVU 2026 Category 2 evaluation metrics.

Loads answer files from metric_testing/ and evaluates them against
the ground truth using ALL 2026 metrics:
  Primary:   BERTScore-F1  (roberta-large, rescaled with baseline)
  Secondary: NLI Entailment (cross-encoder/nli-deberta-v3-base),
             NLI*BERT (nli_entailment * bertscore_f1),
             BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L

Usage:
    python metric_testing/test_metrics.py

    Q1 (case001): In this procedure, what is cauterized by the surgeon?
    Q5 (case005): Which surgical specialty is this procedure associated with?
    Q6 (case006): Are forceps involved in the procedure?
"""
import json
import os
import string
from pathlib import Path

# Use locally cached models (downloaded via download_models_host.py)
_MODEL_CACHE = Path(__file__).resolve().parent.parent / "model_cache"
os.environ["HF_HOME"] = str(_MODEL_CACHE / "huggingface")
os.environ["NLTK_DATA"] = str(_MODEL_CACHE / "nltk_data")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
from bert_score import BERTScorer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import CrossEncoder


# ---------------------------------------------------------------------------
# Metric helpers (same as evaluate.py)
# ---------------------------------------------------------------------------

def normalize(text):
    return text.translate(str.maketrans("", "", string.punctuation)).lower().strip()


def compute_bleu(candidate, references):
    cand_tokens = candidate.split()
    smoothing = SmoothingFunction().method1
    weights = (0.25, 0.25, 0.25, 0.25)
    scores = []
    for ref in references:
        ref_tokens = ref.split()
        s = sentence_bleu(
            [ref_tokens], cand_tokens,
            weights=weights, smoothing_function=smoothing,
        )
        scores.append(s)
    return max(scores) if scores else 0.0


def compute_rouge(candidate, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
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


def compute_bertscore(candidate, references, scorer):
    cands = [candidate] * len(references)
    _P, _R, F1 = scorer.score(cands, references)
    return F1.max().item()


def compute_nli_entailment(candidate, references, model):
    pairs = [(candidate, ref) for ref in references]
    logits = model.predict(pairs)
    # label mapping: 0=contradiction, 1=entailment, 2=neutral
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    entailment_probs = probs[:, 1]
    return float(entailment_probs.max())


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(label, candidate, metrics):
    print(f"  {label}")
    print(f"    Answer: {candidate[:90]}{'…' if len(candidate) > 90 else ''}")
    print(f"    BERTScore-F1:         {metrics['bertscore_f1']:.4f}  (PRIMARY)")
    print(f"    NLI Entailment:       {metrics['nli_entailment']:.4f}")
    print(f"    NLI*BERT:             {metrics['nli_bertscore_f1']:.4f}")
    print(f"    BLEU-4:               {metrics['bleu_score']:.4f}")
    print(f"    ROUGE-1:              {metrics['rouge1_score']:.4f}")
    print(f"    ROUGE-2:              {metrics['rouge2_score']:.4f}")
    print(f"    ROUGE-L:              {metrics['rougeL_score']:.4f}")
    print()


def run_test(candidate, truth_list, bert_scorer, nli_model):
    refs_clean = [normalize(r) for r in truth_list]
    cand_clean = normalize(candidate)

    bleu = compute_bleu(cand_clean, refs_clean)
    rouge = compute_rouge(cand_clean, refs_clean)
    bertscore_f1 = compute_bertscore(candidate, truth_list, bert_scorer)
    nli = compute_nli_entailment(candidate, truth_list, nli_model)
    nli_bert = nli * bertscore_f1

    return {
        "bertscore_f1": bertscore_f1,
        "nli_entailment": nli,
        "nli_bertscore_f1": nli_bert,
        "bleu_score": bleu,
        "rouge1_score": rouge["rouge1"],
        "rouge2_score": rouge["rouge2"],
        "rougeL_score": rouge["rougeL"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    case_list = ["001", "005", "006"]

    script_dir = Path(__file__).parent
    ground_truth_dir = script_dir.parent / "ground_truth"

    # Load heavy models once
    print("Loading BERTScore model (roberta-large) ...")
    bert_scorer = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True)

    print("Loading NLI Cross-Encoder model (cross-encoder/nli-deberta-v3-base) ...")
    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    print()

    for case_num in case_list:
        # Load ground truth
        ground_truth_path = ground_truth_dir / f"case{case_num}.json"
        try:
            with open(ground_truth_path, "r") as f:
                ground_truths = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Ground truth file not found: {ground_truth_path}")
            continue

        print("=" * 70)
        print(f"CASE {case_num}  ({len(ground_truths)} reference answers)")
        print(f"  GT[0]: {ground_truths[0]}")
        print("=" * 70)

        # Find all answer files for this case
        answer_files = sorted(script_dir.glob(f"answer{case_num}*.json"))

        if not answer_files:
            print(f"  WARNING: No answer files found for case {case_num}")
            print(f"  Create answer{case_num}a.json etc. with your test answer.\n")
            continue

        for answer_path in answer_files:
            answer_name = answer_path.stem
            variant = answer_name.replace(f"answer{case_num}", "") or "(base)"

            try:
                with open(answer_path, "r") as f:
                    test_answer = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  ERROR: Invalid JSON in {answer_path.name}: {e}\n")
                continue

            metrics = run_test(test_answer, ground_truths, bert_scorer, nli_model)
            print_results(f"Variant {variant}", test_answer, metrics)

    print("=" * 70)
    print("All tests completed.")


if __name__ == "__main__":
    main()

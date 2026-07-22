#!/usr/bin/env python3
"""
Download all required models LOCALLY on the host machine.

Run this ONCE before `./do_build.sh`. Models are saved to ./model_cache/
which the Dockerfile then COPYs into the image — no in-Docker download needed.

Usage:
    pip install bert-score sentence-transformers nltk
    python download_models_host.py

Requires: HF_TOKEN env var for faster downloads (optional but recommended).
"""

import os
import sys
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "model_cache"
HF_CACHE = CACHE_DIR / "huggingface"
NLTK_CACHE = CACHE_DIR / "nltk_data"

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["NLTK_DATA"] = str(NLTK_CACHE)

print(f"Model cache directory: {CACHE_DIR}")
print(f"  HF_HOME  = {HF_CACHE}")
print(f"  NLTK_DATA = {NLTK_CACHE}")
print()

# --- BERTScore model (roberta-large) ---
from bert_score import BERTScorer

print("Downloading BERTScore model (roberta-large) ...")
scorer = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True)
del scorer
print("  done.\n")

# --- NLI Cross-Encoder model ---
from sentence_transformers import CrossEncoder

print("Downloading NLI Cross-Encoder model (cross-encoder/nli-deberta-v3-base) ...")
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
del nli_model
print("  done.\n")

# --- NLTK data ---
import nltk

print("Downloading NLTK data ...")
NLTK_CACHE.mkdir(parents=True, exist_ok=True)
nltk.download("punkt", download_dir=str(NLTK_CACHE), quiet=True)
nltk.download("punkt_tab", download_dir=str(NLTK_CACHE), quiet=True)
nltk.download("wordnet", download_dir=str(NLTK_CACHE), quiet=True)
print("  done.\n")

# Summary
total_size = sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file())
print(f"All models downloaded to {CACHE_DIR}")
print(f"Total size: {total_size / 1e9:.2f} GB")
print()
print("You can now run:  ./do_build.sh")

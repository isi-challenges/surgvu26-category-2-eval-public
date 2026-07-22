#!/usr/bin/env python3
"""
Generate a full 131-case test input by randomly picking one ground-truth
reference answer per case as the "prediction".

This simulates a near-perfect submission for sanity checking the evaluation
container.  Since the prediction IS one of the references, BERTScore and
most metrics should be very high (close to 1.0).

Usage:
    python generate_test_input.py
    # then: ./do_test_run.sh
"""

import json
import os
import random
import shutil
import uuid
from pathlib import Path

SEED = 42
random.seed(SEED)

SCRIPT_DIR = Path(__file__).parent
GT_DIR = SCRIPT_DIR / "ground_truth"
INPUT_DIR = SCRIPT_DIR / "test" / "input"

# Clean existing test input
if INPUT_DIR.exists():
    shutil.rmtree(INPUT_DIR)
INPUT_DIR.mkdir(parents=True)

gt_files = sorted(GT_DIR.glob("case*.json"))
print(f"Found {len(gt_files)} ground truth files")

predictions_json = []

for gt_file in gt_files:
    with open(gt_file) as f:
        references = json.load(f)

    # Randomly pick one reference as the prediction
    chosen = random.choice(references)
    case_slug = gt_file.stem  # e.g. "case001"

    # Generate a unique job pk
    pk = str(uuid.uuid4())

    # Create the output directory structure: <pk>/output/visual-context-response.json
    output_dir = INPUT_DIR / pk / "output"
    output_dir.mkdir(parents=True)
    with open(output_dir / "visual-context-response.json", "w") as f:
        json.dump(chosen, f)

    # Build the predictions.json entry
    # The input slug must match the ground truth filename (without .json)
    predictions_json.append({
        "pk": pk,
        "inputs": [
            {
                "socket": {
                    "slug": "endoscopic-robotic-surgery-video",
                    "relative_path": "endoscopic-robotic-surgery-video.mp4",
                    "is_image_kind": False,
                    "is_panimg_kind": False,
                    "is_dicom_image_kind": False,
                    "is_json_kind": False,
                    "is_file_kind": True,
                },
                "file": f"https://grand-challenge.org/media/some-link/{case_slug}.mp4",
                "image": None,
                "value": None,
            },
            {
                "socket": {
                    "slug": "visual-context-question",
                    "relative_path": "visual-context-question.json",
                    "example_value": "Example String",
                    "is_image_kind": False,
                    "is_panimg_kind": False,
                    "is_dicom_image_kind": False,
                    "is_json_kind": True,
                    "is_file_kind": False,
                },
                "file": None,
                "image": None,
                "value": "Example String",
            },
        ],
        "outputs": [
            {
                "socket": {
                    "slug": "visual-context-response",
                    "relative_path": "visual-context-response.json",
                    "example_value": "Example String",
                    "is_image_kind": False,
                    "is_panimg_kind": False,
                    "is_dicom_image_kind": False,
                    "is_json_kind": True,
                    "is_file_kind": False,
                },
                "file": None,
                "image": None,
                "value": chosen,
            },
        ],
        "exec_duration": "PT1M0S",
        "invoke_duration": None,
        "status": "Succeeded",
    })

    print(f"  {case_slug} (pk={pk[:8]}…) → \"{chosen[:60]}{'…' if len(chosen)>60 else ''}\"")

# Write predictions.json
with open(INPUT_DIR / "predictions.json", "w") as f:
    json.dump(predictions_json, f, indent=4)

print(f"\nGenerated {len(predictions_json)} test cases in {INPUT_DIR}")
print("Run ./do_test_run.sh to evaluate")

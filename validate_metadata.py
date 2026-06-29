#!/usr/bin/env python3
"""Validate figure_metadata.json — check structure, categories, prompts.

Usage:
    python3 validate_metadata.py <path_to_figure_metadata.json>
"""

import sys
import json
from pathlib import Path
from collections import Counter


def validate(metadata_path: str) -> bool:
    path = Path(metadata_path)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return False

    with open(path) as f:
        data = json.load(f)

    errors = []

    # Check top-level keys
    for key in ["book_context", "category_prompts", "figures"]:
        if key not in data:
            errors.append(f"Missing top-level key: '{key}'")

    # Check figures
    figures = data.get("figures", [])
    if not figures:
        print("WARNING: No figures in metadata")

    cats = Counter()
    missing_caption = 0
    missing_prompt = 0

    for i, fig in enumerate(figures):
        prefix = f"Figure[{i}] ({fig.get('filename', '?')}):"
        for field in ["filename", "name", "caption", "category", "prompt"]:
            if field not in fig:
                errors.append(f"{prefix} missing '{field}'")
            elif not fig[field]:
                if field == "caption":
                    missing_caption += 1
                elif field == "prompt":
                    missing_prompt += 1
                    errors.append(f"{prefix} empty prompt")
        cats[fig.get("category", "unknown")] += 1

    # Check that every category has a prompt in category_prompts
    prompts = data.get("category_prompts", {})
    unique_cats = set(cats.keys())
    for cat in unique_cats:
        if cat not in prompts:
            errors.append(f"Category '{cat}' has no entry in category_prompts")

    # Print results
    print(f"Book: {data.get('book_context', 'N/A')}")
    print(f"Total figures: {len(figures)}")
    print(f"Categories: {len(unique_cats)}")
    print()

    print("Category distribution:")
    for cat, count in cats.most_common():
        has_prompt = "\u2705" if cat in prompts else "\u274c"
        print(f"  {has_prompt} {cat}: {count}")

    print()
    if missing_caption:
        print(f"Figures with empty caption: {missing_caption}")
    if missing_prompt:
        print(f"Figures with empty prompt: {missing_prompt}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"  \u274c {e}")
        return False
    else:
        print("\u2705 All validations passed!")
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 validate_metadata.py <path_to_figure_metadata.json>")
        sys.exit(1)
    ok = validate(sys.argv[1])
    sys.exit(0 if ok else 1)

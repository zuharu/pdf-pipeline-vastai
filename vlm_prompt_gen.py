#!/usr/bin/env python3
"""
vlm_prompt_gen.py — Generate prompt_config.json from book markdown via DeepSeek API.

Phase 2 of PDF Ingestion Pipeline (runs on Vast.ai GPU instance):
  Takes a book's markdown output, analyzes figure patterns via DeepSeek LLM,
  generates custom VLM prompt templates stored in prompt_config.json.

Usage:
    python3 /workspace/vlm_prompt_gen.py /workspace/staging/<bookname>/
    python3 /workspace/vlm_prompt_gen.py /workspace/staging/<bookname>/ --model deepseek-v4-pro

API: OpenAI SDK → DeepSeek (OpenAI-compatible, base_url="https://api.deepseek.com")
Ref:  https://api-docs.deepseek.com — reasoning_effort="high", thinking type "enabled"
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DEEPSEEK_MODEL = "deepseek-v4-flash"  # $0.14/1M input, $0.28/1M output
MAX_CHARS = 50000  # max markdown to sample for LLM analysis


# ── API Key Loader ────────────────────────────────────────────────────────────
def load_api_key(env_var: str = "DEEPSEEK_API_KEY") -> str:
    """Load API key from environment or .env file."""
    key = os.environ.get(env_var, "")
    if key:
        return key
    # Try .env in workspace (SCP'd by agent from .env.deepseek)
    for env_path in [Path("/workspace/.env"), Path(".env")]:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith(f"{env_var}="):
                        val = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                        return val
    return ""


# ── Markdown Sampler ──────────────────────────────────────────────────────────
def sample_markdown(md_path: str, max_chars: int = MAX_CHARS) -> str:
    """Smart sampling of book markdown for LLM analysis."""
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    parts = []

    # 1. First portion — captures book opening + early chapter patterns
    early = lines[:min(800, total)]
    parts.append("".join(early))

    # 2. Section headers from remainder — structural index
    mid_sections = []
    for line in lines[800:]:
        stripped = line.lstrip()
        if stripped.startswith("#") and not stripped.startswith("####"):
            mid_sections.append(stripped.rstrip())
    if mid_sections:
        parts.append("\n## Section Index (remainder of book):\n")
        parts.append("\n".join(mid_sections[:200]))

    # 3. Distributed figure captions — diversity sampling
    fig_captions = []
    for i, line in enumerate(lines):
        if line.startswith("![") and "(_page_" in line:
            for j in range(i + 1, min(i + 4, len(lines))):
                ll = lines[j].strip()
                if ll and not ll.startswith("![]") and not ll.startswith("|"):
                    fig_captions.append(ll)
                    break

    # Subsample for coverage
    if len(fig_captions) > 100:
        step = max(1, len(fig_captions) // 50)
        fig_captions = fig_captions[::step]
    if fig_captions:
        parts.append(f"\n## Figure Captions (sample of {len(fig_captions)}):\n")
        parts.append("\n".join(fig_captions[:100]))

    return "\n".join(parts)[:max_chars]


# ── Prompt Generation ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert in technical document analysis and figure description. 
Your task is to analyze a book's markdown content and generate optimal VLM (Vision Language Model) 
prompts for describing different categories of figures (circuit schematics, graphs, block diagrams, photos).

Output ONLY valid JSON with this exact structure:
{
  "circuit": {
    "description": "What this category covers",
    "prompt": "Detailed VLM prompt for describing circuit schematic figures. Include what to look for: components (resistors, capacitors, transistors, op-amps), topology, signal flow, labels, values. Output format: structured markdown with ## Components, ## Topology, ## Operation sections."
  },
  "graph": {
    "description": "What this category covers",
    "prompt": "Detailed VLM prompt for describing graphs/plots. Include: axis labels, units, curve shapes, key points, trends, what the graph demonstrates."
  },
  "block_diagram": {
    "description": "What this category covers",
    "prompt": "Detailed VLM prompt for block diagrams. Include: functional blocks, interconnections, signal flow direction, labels, system-level function."
  },
  "photo": {
    "description": "What this category covers", 
    "prompt": "Detailed VLM prompt for photographs. Include: what is shown, equipment/apparatus, scale if visible, context in the chapter."
  }
}

Make each prompt specific to THIS book's content. Look at the chapter topics and figure captions to customize. 
Prompts should be detailed enough that a VLM produces consistent, structured output. 
Each prompt should be 3-5 sentences, technical but clear."""


def generate_prompts(markdown_sample: str, model: str, api_key: str) -> dict:
    """Call DeepSeek API to generate figure description prompts."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this book content and generate figure description prompts:\n\n{markdown_sample}"}
        ],
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        temperature=0.3,
        max_tokens=4096
    )

    content = response.choices[0].message.content.strip()

    # Extract JSON from response (may be wrapped in ```json)
    if "```" in content:
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)

    return json.loads(content)


# ── Validation ────────────────────────────────────────────────────────────────
REQUIRED_KEYS = {"circuit", "graph", "block_diagram", "photo"}


def validate_config(config: dict) -> list[str]:
    """Validate prompt_config.json structure. Returns list of warnings."""
    warnings = []
    missing = REQUIRED_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Missing required categories: {missing}")

    for cat in REQUIRED_KEYS:
        cat_data = config.get(cat, {})
        if not isinstance(cat_data, dict):
            warnings.append(f"{cat}: not a dict")
        if "prompt" not in cat_data:
            warnings.append(f"{cat}: missing 'prompt' field")

    return warnings


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate VLM prompts via DeepSeek API")
    parser.add_argument("book_dir", help="Path to book staging directory (contains <book>.md)")
    parser.add_argument("--model", default=DEEPSEEK_MODEL, help=f"DeepSeek model (default: {DEEPSEEK_MODEL})")
    parser.add_argument("--dry-run", action="store_true", help="Validate without API call")
    args = parser.parse_args()

    book_dir = Path(args.book_dir)
    if not book_dir.is_dir():
        print(f"ERROR: Directory not found: {book_dir}")
        sys.exit(1)

    # Find the markdown file
    md_files = list(book_dir.glob("*.md"))
    if not md_files:
        print(f"ERROR: No .md file found in {book_dir}")
        sys.exit(1)
    md_path = str(md_files[0])
    print(f"[PROMPT GEN] Book: {md_path}")

    # Sample markdown
    sample = sample_markdown(md_path)
    print(f"[PROMPT GEN] Sampled {len(sample):,} chars from markdown")

    if args.dry_run:
        print("[PROMPT GEN] Dry run — skipping API call")
        sys.exit(0)

    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set. Set via env var or /workspace/.env")
        sys.exit(1)

    # Generate prompts
    print(f"[PROMPT GEN] Calling DeepSeek API ({args.model})...")
    try:
        config = generate_prompts(sample, args.model, api_key)
    except Exception as e:
        print(f"ERROR: DeepSeek API call failed: {e}")
        sys.exit(1)

    # Validate
    warnings = validate_config(config)
    for w in warnings:
        print(f"[PROMPT GEN] ⚠️  {w}")

    # Save
    output_path = book_dir / "prompt_config.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[PROMPT GEN] ✅ Saved {output_path} ({len(json.dumps(config))} bytes)")

    # Print summary
    for cat in sorted(config.keys()):
        desc = config[cat].get("description", "no description")
        prompt_len = len(config[cat].get("prompt", ""))
        print(f"  {cat}: {desc} (prompt: {prompt_len} chars)")


if __name__ == "__main__":
    main()
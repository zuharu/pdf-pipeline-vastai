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


# ── Figure Extraction ──────────────────────────────────────────────────────────
def extract_figures_with_context(
    md_path: str,
    lines_above: int = 5,
    lines_below: int = 10,
    max_context_chars: int = 300
) -> list[dict]:
    """
    Extract all figure references from markdown with captions and surrounding text.

    For each ![](_page_*...jpeg) reference:
    - Captures the alt text (usually empty from marker-pdf)
    - Finds the caption: next 1-3 non-empty, non-image lines after the reference
    - Captures N lines above and M lines below as context (truncated to max_context_chars)

    Returns list of {filename, name, alt_text, caption, context}.
    """
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    figures = []
    total = len(lines)

    for i, line in enumerate(lines):
        match = re.match(
            r'!\[(.*?)\]\((_page_\d+_[^.]+\.[jJ][pP][eE]?[gG])\)',
            line.strip()
        )
        if not match:
            continue

        alt_text = match.group(1).strip()
        filename = match.group(2)
        name = re.sub(r'\.[jJ][pP][eE]?[gG]$', '', filename)

        # Extract caption: next 1-3 non-empty, non-image lines
        caption = ""
        for j in range(i + 1, min(i + 4, total)):
            ll = lines[j].strip()
            if ll and not ll.startswith("![]") and not ll.startswith("|"):
                caption = ll
                break

        # Extract context above
        above_start = max(0, i - lines_above)
        context_above = "".join(lines[above_start:i]).strip()

        # Extract context below
        below_end = min(total, i + 1 + lines_below)
        context_below = "".join(lines[i + 1:below_end]).strip()

        # Combine and truncate context
        full_context = (context_above + "\n" + context_below).strip()
        if len(full_context) > max_context_chars:
            full_context = full_context[:150] + "\n...\n" + full_context[-150:]

        figures.append({
            "filename": filename,
            "name": name,
            "alt_text": alt_text,
            "caption": caption,
            "context": full_context
        })

    return figures


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


CLASSIFY_FIGURES_SYSTEM_PROMPT = """You are an expert in classifying scientific and technical figures.

Your task: classify each figure into one of these categories based on its filename, caption, and surrounding context.

Categories:
- **circuit**: Electronic circuit schematics, wiring diagrams, component-level diagrams. Shows resistors, capacitors, transistors, op-amps, signal paths.
- **graph**: Data plots, charts, frequency response curves, Bode plots, noise spectra. Has axes, curves, data points.
- **block_diagram**: High-level system diagrams, functional blocks, signal flow diagrams, architecture diagrams. Shows boxes connected by arrows.
- **photo**: Photographs of equipment, lab setups, physical objects, people, micrographs.
- **other**: If the figure doesn't fit the above, use "other" with a brief subtype like:
    other_micrograph (microscope images of cells/tissue)
    other_table (data tables rendered as images)
    other_illustration (artistic or conceptual drawings)
    other_screenshot (software screenshots)
    other_unknown (cannot determine from context)

Rules:
1. Use the caption FIRST — it usually contains keywords like "circuit", "schematic", "frequency response", "block diagram", "photograph"
2. If the caption is empty or ambiguous, use the surrounding context
3. The filename (e.g., _page_31_Figure_6) does NOT contain useful classification info — ignore it
4. For "other" figures, ALWAYS include a subtype like other_micrograph, other_table, etc.
5. If truly uncertain, use "other_unknown"

Output ONLY valid JSON with this EXACT structure:
{
  "figures": [
    {"filename": "string", "category": "string"},
    ...
  ]
}"""


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


# ── Figure Classification ──────────────────────────────────────────────────────
def classify_figures_batch(
    figures: list[dict],
    book_context: str,
    model: str,
    api_key: str,
    max_figures_per_call: int = 500
) -> list[dict]:
    """Mass-classify all figures via DeepSeek API (single call or batched)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    all_classified = []

    for batch_start in range(0, len(figures), max_figures_per_call):
        batch = figures[batch_start:batch_start + max_figures_per_call]
        figure_list_text = []
        for fig in batch:
            entry = f"File: {fig['filename']}\n"
            if fig['caption']:
                entry += f"Caption: {fig['caption']}\n"
            if fig['context']:
                entry += f"Context: {fig['context']}\n"
            figure_list_text.append(entry)

        all_entries = "\n---\n".join(figure_list_text)
        user_prompt = f"""Book: {book_context}

Classify each of the following {len(batch)} figures:

{all_entries}
"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CLASSIFY_FIGURES_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
                temperature=0.1,
                max_tokens=8192
            )
        except Exception as e:
            raise RuntimeError(f"DeepSeek classification API call failed: {e}") from e

        content = response.choices[0].message.content.strip()
        if "```" in content:
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```\s*", "", content)

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"DeepSeek returned invalid JSON: {e}\nFirst 500 chars: {content[:500]}") from e

        if "figures" not in result:
            raise RuntimeError(f"Response missing 'figures' key. Keys: {list(result.keys())}")

        classified_map = {item["filename"]: item["category"] for item in result["figures"]}
        for fig in batch:
            fig["category"] = classified_map.get(fig["filename"], "other_unknown")
            all_classified.append(fig)

    return all_classified


# ── Other Prompts Generation ───────────────────────────────────────────────────
OTHER_PROMPTS_SYSTEM_PROMPT = """You are an expert at generating VLM (Vision Language Model) description prompts for scientific figures.

Given a list of "other" figure subtypes (e.g., other_micrograph, other_table), generate a detailed VLM description prompt for EACH subtype.

Each prompt should be 3-5 sentences instructing the VLM what to look for and how to structure the output.

For the special subtype "other_unknown" (figures we cannot identify), generate a GENERIC fallback prompt that works for ANY type of figure.

Output ONLY valid JSON:
{
  "prompts": {
    "other_micrograph": {
      "description": "Microscope images of biological or material specimens",
      "prompt": "Describe the cellular structures or material features visible..."
    },
    "other_unknown": {
      "description": "Generic fallback for unidentifiable figures",
      "prompt": "Describe this figure in detail. Identify what is shown..."
    }
  }
}"""


def generate_other_prompts(figures, book_context, model, api_key):
    """Generate custom VLM prompts for 'other' figure subtypes."""
    subtypes = set()
    for fig in figures:
        cat = fig.get("category", "")
        if cat.startswith("other_"):
            subtypes.add(cat)
    subtypes.add("other_unknown")

    subtype_examples = {}
    for subtype in subtypes:
        examples = []
        for fig in figures:
            if fig.get("category") == subtype:
                example = f"File: {fig['filename']}\n"
                if fig.get("caption"):
                    example += f"Caption: {fig['caption']}\n"
                if fig.get("context"):
                    example += f"Context: {fig['context'][:200]}\n"
                examples.append(example)
                if len(examples) >= 5:
                    break
        subtype_examples[subtype] = examples

    # Build prompt
    subtypes_list = "\n".join(f"- {s}: {len(subtype_examples.get(s, []))} examples" for s in sorted(subtypes))
    examples_text = ""
    for subtype, ex_list in sorted(subtype_examples.items()):
        if ex_list:
            examples_text += f"\n### {subtype} examples:\n"
            examples_text += "\n---\n".join(ex_list[:3])

    user_prompt = f"""Book: {book_context}

Generate VLM description prompts for these "other" figure subtypes:

{subtypes_list}

Here are example figures for context:
{examples_text if examples_text else '(no examples available)'}

Generate a detailed VLM prompt for EACH subtype listed above.
"""

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": OTHER_PROMPTS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            reasoning_effort="medium",
            extra_body={"thinking": {"type": "enabled"}},
            temperature=0.3,
            max_tokens=4096
        )
    except Exception as e:
        print(f"WARN: Other prompts generation failed: {e}")
        return _hardcoded_other_prompts(subtypes)

    content = response.choices[0].message.content.strip()
    if "```" in content:
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)
    try:
        result = json.loads(content)
        return result.get("prompts", _hardcoded_other_prompts(subtypes))
    except json.JSONDecodeError:
        return _hardcoded_other_prompts(subtypes)


def _hardcoded_other_prompts(subtypes):
    """Hardcoded fallback prompts for 'other' figure subtypes."""
    fallback = {
        "other_unknown": {
            "description": "Generic fallback for unidentifiable figures",
            "prompt": "Describe this figure in detail. Identify what is shown — whether it's a diagram, photograph, table, or illustration. Note any visible labels, numbers, or annotations. Explain what the figure demonstrates based on surrounding context. Output in structured markdown."
        },
        "other_micrograph": {
            "description": "Microscope images",
            "prompt": "Describe the cellular structures or material features visible in this micrograph. Note the magnification if visible. Identify any staining or imaging technique that may have been used. Describe the key features: shape, size, arrangement, any anomalies."
        },
        "other_table": {
            "description": "Data tables rendered as images",
            "prompt": "Extract and describe the data presented in this table. List the column headers and row labels. Summarize the key numerical values, trends, or comparisons shown."
        },
        "other_screenshot": {
            "description": "Software screenshots",
            "prompt": "Describe the software interface shown in this screenshot. Identify the application if possible, the visible UI elements (menus, buttons, panels), and what operation or result is displayed."
        }
    }
    result = {}
    for s in subtypes:
        result[s] = fallback.get(s, fallback["other_unknown"])
    return result


# ── Metadata Output ────────────────────────────────────────────────────────────
def save_figure_metadata(figures, category_prompts, book_context, output_path):
    """Write figure_metadata.json — the data contract between Phase B and Phase C."""
    figure_entries = []
    for fig in figures:
        cat = fig.get("category", "other_unknown")
        cat_info = category_prompts.get(cat, category_prompts.get("other_unknown", {}))
        prompt = cat_info.get("prompt", "Describe this figure in detail.")
        figure_entries.append({
            "filename": fig["filename"],
            "name": fig["name"],
            "caption": fig.get("caption", ""),
            "category": cat,
            "prompt": prompt
        })

    metadata = {
        "book_context": book_context,
        "category_prompts": category_prompts,
        "figures": figure_entries
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    cat_counts = {}
    for fig in figure_entries:
        cat = fig["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print(f"[METADATA] Saved {output_path}")
    print(f"[METADATA] {len(figure_entries)} figures:")
    for cat in sorted(cat_counts.keys()):
        print(f"  {cat}: {cat_counts[cat]}")


# ── Validation ────────────────────────────────────────────────────────────────
def validate_config(config: dict) -> list[str]:
    """Validate prompt_config.json structure. Returns list of warnings."""
    warnings = []
    if not isinstance(config, dict):
        raise ValueError(f"Config is not a dict: {type(config)}")
    for cat, cat_data in config.items():
        if not isinstance(cat_data, dict):
            warnings.append(f"{cat}: not a dict")
            continue
        if "prompt" not in cat_data:
            warnings.append(f"{cat}: missing 'prompt' field")
    return warnings


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate VLM prompts + classify figures via DeepSeek API")
    parser.add_argument("book_dir", help="Path to book staging directory (contains <book>.md)")
    parser.add_argument("--model", default=DEEPSEEK_MODEL, help=f"DeepSeek model (default: {DEEPSEEK_MODEL})")
    parser.add_argument("--dry-run", action="store_true", help="Validate without API call")
    parser.add_argument("--skip-classify", action="store_true",
                       help="Skip figure classification (only generate prompts)")
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
    book_name = Path(md_path).stem
    print(f"[PROMPT GEN] Book: {book_name} ({md_path})")

    # Sample markdown for book context
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

    # ── Existing: Generate category-level prompts ──
    print(f"[PROMPT GEN] Calling DeepSeek API ({args.model}) for prompt generation...")
    try:
        config = generate_prompts(sample, args.model, api_key)
    except json.JSONDecodeError as e:
        print(f"ERROR: DeepSeek returned invalid JSON for prompts: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"ERROR: DeepSeek API call failed: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)

    # Validate existing categories
    warnings = validate_config(config)
    for w in warnings:
        print(f"[PROMPT GEN] ⚠️  {w}")

    # ── NEW: Extract figures with context ──
    print(f"[PROMPT GEN] Extracting figures from markdown...")
    try:
        figures = extract_figures_with_context(md_path)
    except Exception as e:
        print(f"ERROR: Figure extraction failed: {e}")
        sys.exit(1)

    print(f"[PROMPT GEN] Found {len(figures)} figure references")

    if len(figures) == 0:
        print("[PROMPT GEN] ⚠️  No figures found — skipping classification")
        args.skip_classify = True

    # ── NEW: Mass classify figures via DeepSeek ──
    if not args.skip_classify and len(figures) > 0:
        book_context = book_name.replace("_", " ")
        print(f"[PROMPT GEN] Classifying {len(figures)} figures via DeepSeek {args.model}...")
        try:
            figures = classify_figures_batch(figures, book_context, args.model, api_key)
        except Exception as e:
            print(f"ERROR: Figure classification failed: {e}")
            sys.exit(1)

        # Print classification summary
        cat_counts = {}
        for fig in figures:
            cat = fig.get("category", "other_unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        print(f"[PROMPT GEN] Classification results:")
        for cat in sorted(cat_counts.keys()):
            print(f"  {cat}: {cat_counts[cat]}")

        # ── NEW: Generate prompts for "other" subtypes ──
        other_subtypes = {f.get("category") for f in figures
                         if f.get("category", "").startswith("other_")}

        # Filter to non-unknown subtypes
        custom_others = other_subtypes - {"other_unknown"}
        if custom_others:
            print(f"[PROMPT GEN] Generating prompts for {len(custom_others)} 'other' subtypes...")
            try:
                other_prompts = generate_other_prompts(
                    figures, book_context, args.model, api_key
                )
                # Merge into config
                for subtype, prompt_data in other_prompts.items():
                    if subtype not in config:
                        config[subtype] = prompt_data
            except Exception as e:
                print(f"WARN: Other prompt generation failed: {e}")
                print("Using hardcoded fallback for other subtypes.")
                for subtype in custom_others:
                    if subtype not in config:
                        config[subtype] = {
                            "description": f"Auto-generated for {subtype}",
                            "prompt": "Describe this figure in detail. Identify what is shown."
                        }
        else:
            print("[PROMPT GEN] No custom 'other' subtypes — using standard categories only")

    # Ensure other_unknown always exists
    if "other_unknown" not in config:
        config["other_unknown"] = {
            "description": "Generic fallback for unidentifiable figures",
            "prompt": (
                "Describe this figure in detail. Identify what is shown — "
                "whether it's a diagram, photograph, table, or illustration. "
                "Note any visible labels, numbers, or annotations."
            )
        }

    # ── Save prompt_config.json (backward compat) ──
    pconfig_path = book_dir / "prompt_config.json"
    with open(pconfig_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[PROMPT GEN] ✅ Saved {pconfig_path} ({len(json.dumps(config))} bytes)")

    # Print prompt summary
    for cat in sorted(config.keys()):
        desc = config[cat].get("description", "no description")
        prompt_len = len(config[cat].get("prompt", ""))
        print(f"  {cat}: {desc} (prompt: {prompt_len} chars)")

    # ── NEW: Save figure_metadata.json ──
    # Always write figure_metadata.json so Phase C can run (even if 0 figures).
    # When classification is skipped, write minimal metadata from what we have.
    if args.skip_classify or len(figures) == 0:
        # Write minimal metadata — no per-figure classification, but Phase C can start
        # (it will find 0 figures and exit cleanly)
        minimal_figures = [] if args.skip_classify else [
            {"filename": f["filename"], "name": f["name"],
             "caption": f.get("caption", ""), "category": "other_unknown",
             "prompt": config.get("other_unknown", {}).get("prompt", "Describe this figure.")}
            for f in figures
        ] if len(figures) > 0 else []
        metadata_path = book_dir / "figure_metadata.json"
        save_figure_metadata(
            minimal_figures if minimal_figures else [],
            config, book_context, metadata_path
        )
        if args.skip_classify:
            print("[PROMPT GEN] ⚠️  Classification skipped — figure_metadata.json has no per-figure categories")
        else:
            print("[PROMPT GEN] ⚠️  No figures found — empty figure_metadata.json written")
    else:
        metadata_path = book_dir / "figure_metadata.json"
        save_figure_metadata(figures, config, book_context, metadata_path)

    # Phase C can now run with figure_metadata.json regardless of classification path
    print("[PROMPT GEN] ✅ Phase B complete — Phase C can now run with figure_metadata.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
vlm_describe.py — Describe figures via self-hosted Ollama VLM.

Phase 3 of PDF Ingestion Pipeline (runs on Vast.ai GPU instance):
  Takes figure_metadata.json + figure JPEGs, processes each figure
  via Ollama (Qwen2.5-VL-7B-Instruct), saves results, assembles <book>_enhanced.md.

Resumable — state saved to .vlm_job/ in the staging directory.

Requires: figure_metadata.json from vlm_prompt_gen.py (Phase B)

Usage:
    python3 /workspace/vlm_describe.py /workspace/staging/<bookname>/
    python3 /workspace/vlm_describe.py /workspace/staging/<bookname>/ --max-workers 4
    python3 /workspace/vlm_describe.py /workspace/staging/<bookname>/ --status

API: ollama Python client — ollama.chat(model, messages=[{images: [...]}])
Ref:  https://context7.com/ollama/ollama-python — Vision API with images parameter
"""

import re
import sys
import json
import argparse
import base64
from pathlib import Path
from datetime import datetime

# ── Rich ──────────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TaskProgressColumn,
    TimeRemainingColumn, SpinnerColumn,
)

console = Console()

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_MODEL = "qwen2.5vl:7b"  # Qwen2.5-VL-7B-Instruct (Q4_K_M, 6.5 GB VRAM)
DEFAULT_MAX_WORKERS = 2         # Conservative — leave VRAM headroom


# ── State Management ──────────────────────────────────────────────────────────
def load_state(job_dir: Path) -> dict:
    """Load resume state, return empty if none."""
    state_file = job_dir / "state.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"processed": {}, "errors": {}, "started_at": None, "updated_at": None}


def save_state(job_dir: Path, state: dict):
    state["updated_at"] = str(datetime.now())
    with open(job_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2)


def load_errors(job_dir: Path) -> dict:
    err_file = job_dir / "errors.json"
    if err_file.exists():
        with open(err_file) as f:
            return json.load(f)
    return {}


def save_error(job_dir: Path, figure_name: str, error: str):
    errors = load_errors(job_dir)
    errors[figure_name] = {"error": str(error), "timestamp": str(datetime.now())}
    with open(job_dir / "errors.json", "w") as f:
        json.dump(errors, f, indent=2)


# ── Figure Metadata Loading ────────────────────────────────────────────────────
def load_figure_metadata(book_dir: Path) -> dict:
    """
    Load figure_metadata.json — the data contract from Phase B.

    Returns: {
        "book_context": str,
        "category_prompts": dict,
        "figures": [{"filename", "name", "caption", "category", "prompt", "path"}, ...]
    }

    Errors if file not found or malformed.
    """
    metadata_path = book_dir / "figure_metadata.json"
    if not metadata_path.exists():
        console.print(
            f"[red]ERROR:[/] figure_metadata.json not found in {book_dir}\n"
            f"Run vlm_prompt_gen.py (Phase B) first to generate it."
        )
        sys.exit(1)

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    # Validate structure
    if "figures" not in metadata:
        console.print("[red]ERROR:[/] figure_metadata.json missing 'figures' key")
        sys.exit(1)

    # Validate each figure entry has required fields
    for fig in metadata["figures"]:
        if "name" not in fig:
            console.print(f"[red]ERROR:[/] Figure entry missing 'name': {fig}")
            sys.exit(1)
        if "category" not in fig:
            fig["category"] = "other_unknown"
        if "prompt" not in fig:
            fig["prompt"] = "Describe this figure in detail."

    # Build path lookup: map figure name to JPEG on disk
    jpegs = {p.name: str(p) for p in book_dir.glob("_page_*.jp*g")}
    missing = 0
    for fig in metadata["figures"]:
        filename = fig.get("filename", "")
        if filename in jpegs:
            fig["path"] = jpegs[filename]
        else:
            # Try alternative extensions
            name = fig["name"]
            found = False
            for ext in [".jpeg", ".jpg", ".JPEG", ".JPG"]:
                if name + ext in jpegs:
                    fig["path"] = jpegs[name + ext]
                    fig["filename"] = name + ext
                    found = True
                    break
            if not found:
                missing += 1

    if missing > 0:
        console.print(f"[yellow]WARN:[/] {missing} figures have no JPEG on disk")

    return metadata

# ── Image Encoding ────────────────────────────────────────────────────────────
def encode_image(image_path: Path) -> str:
    """Read image file, return base64 encoded string for Ollama vision API."""
    return base64.b64encode(image_path.read_bytes()).decode()


# ── VLM Description ───────────────────────────────────────────────────────────
def describe_figure(
    image_path: Path,
    category: str,
    prompt: str,
    model: str = OLLAMA_MODEL
) -> str:
    """
    Call Ollama vision API to describe a single figure.

    Args:
        image_path: Path to JPEG image
        category: Figure category (circuit, graph, block_diagram, photo)
        prompt: VLM prompt for this category from figure_metadata.json
        model: Ollama model tag
    Returns:
        Markdown description string
    """
    from ollama import chat

    image_b64 = encode_image(image_path)

    try:
        response = chat(
            model=model,
            messages=[{
                "role": "user",
                "content": f"{prompt}\n\nDescribe this {category} figure in detail. Output in structured markdown.",
                "images": [image_b64]
            }],
            options={
                "temperature": 0.3,
                "num_predict": 1024
            }
        )
        return response.message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Ollama chat failed for {image_path.name}: {e}") from e


# ── Assembly ──────────────────────────────────────────────────────────────────
def assemble_enhanced_markdown(
    md_path: Path,
    figures: list[dict],
    descriptions: dict[str, str],
    output_path: Path
):
    """Insert VLM descriptions into markdown, produce _enhanced.md."""
    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    result_lines = []
    i = 0
    inserted = 0

    while i < len(lines):
        line = lines[i]
        result_lines.append(line)

        # Detect figure reference — matches marker-pdf format
        # e.g. ![](_page_3_Figure_8.jpeg) or ![](_page_12_Picture_2.jpeg)
        match = re.match(r'!\[.*?\]\((_page_\d+_[^.]+\.[jJ][pP][eE]?[gG])\)', line)
        if match:
            filename = match.group(1)
            fig_name = filename.replace(".jpeg", "").replace(".jpg", "")
            if fig_name in descriptions:
                desc = descriptions[fig_name]
                # Find matching figure metadata
                fig_meta = next((f for f in figures if f["name"] == fig_name), None)
                cat = fig_meta["category"] if fig_meta else "unknown"

                result_lines.append("")
                result_lines.append("<details>")
                result_lines.append(f"<summary><b>🤖 VLM Description</b> ({cat})</summary>")
                result_lines.append("")
                result_lines.append(desc)
                result_lines.append("")
                result_lines.append("</details>")
                inserted += 1

        i += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))

    return inserted


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Describe figures via Ollama VLM")
    parser.add_argument("book_dir", help="Path to book staging directory")
    parser.add_argument("--model", default=OLLAMA_MODEL, help=f"Ollama model (default: {OLLAMA_MODEL})")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                       help=f"Max parallel workers (default: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--status", action="store_true", help="Show job status and exit")
    parser.add_argument("--sample", type=int, default=0, help="Process only N figures (test mode)")
    parser.add_argument("--force", action="store_true", help="Reprocess already-done figures")
    args = parser.parse_args()

    book_dir = Path(args.book_dir)
    if not book_dir.is_dir():
        console.print(f"[red]ERROR:[/] Directory not found: {book_dir}")
        sys.exit(1)

    # Find markdown
    md_files = list(book_dir.glob("*.md"))
    if not md_files:
        console.print(f"[red]ERROR:[/] No .md file found in {book_dir}")
        sys.exit(1)
    md_path = md_files[0]
    book_name = md_path.stem

    # Setup job directory
    job_dir = book_dir / ".vlm_job"
    job_dir.mkdir(exist_ok=True)

    # Load figure metadata from Phase B (replaces discover_figures + classify_figures)
    metadata = load_figure_metadata(book_dir)
    figures = metadata["figures"]
    category_prompts = metadata.get("category_prompts", {})

    # Print category distribution
    cat_counts = {}
    for fig in figures:
        cat = fig.get("category", "other_unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    console.print(f"[VLM] Found {len(figures)} figures in {book_name}:")
    for cat in sorted(cat_counts.keys()):
        console.print(f"  {cat}: {cat_counts[cat]}")

    if args.status:
        state = load_state(job_dir)
        errors = load_errors(job_dir)
        console.print(f"[VLM] Processed: {len(state.get('processed', {}))}, Errors: {len(errors)}")
        return

    # Apply sample limit
    if args.sample > 0:
        figures = figures[:args.sample]
        console.print(f"[VLM] Sample mode: {len(figures)} figures")

    # Load state
    state = load_state(job_dir)
    if not state.get("started_at"):
        state["started_at"] = str(datetime.now())
    save_state(job_dir, state)

    # Filter already-processed
    pending = []
    for fig in figures:
        if not args.force and fig["name"] in state.get("processed", {}):
            continue
        cat = fig.get("category", "other_unknown")
        prompt = fig.get("prompt", "")
        if not prompt:
            # Fallback: use category_prompts from metadata
            prompt = category_prompts.get(cat, {}).get("prompt", "")
        if not prompt:
            console.print(f"[yellow]WARN:[/] No prompt for '{fig['name']}' (category: {cat}), skipping")
            continue
        # Store prompt on the figure dict for process_one() to use
        fig["_prompt"] = prompt
        pending.append(fig)

    if not pending:
        console.print(f"[green]✅ All {len(figures)} figures already processed.[/]")
    else:
        console.print(f"[VLM] Processing {len(pending)} pending figures "
                      f"(model: {args.model}, workers: {args.max_workers})")

    # Process
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    lock = threading.Lock()
    descriptions = {}
    errors = 0
    rate_limit_sem = threading.Semaphore(args.max_workers)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Describing {book_name}...", total=len(pending))

        def process_one(fig):
            nonlocal errors
            rate_limit_sem.acquire()
            try:
                cat = fig["category"]
                prompt = fig.get("_prompt", "")
                desc = describe_figure(
                    Path(fig["path"]),
                    cat,
                    prompt,
                    model=args.model
                )
                with lock:
                    descriptions[fig["name"]] = desc
                    state["processed"][fig["name"]] = {
                        "category": cat,
                        "timestamp": str(datetime.now()),
                        "chars": len(desc),
                        "content": desc  # store full content for resume
                    }
                    save_state(job_dir, state)
            except Exception as e:
                with lock:
                    errors += 1
                    save_error(job_dir, fig["name"], str(e))
                    console.print(f"[red]ERROR {fig['name']}:[/] {e}")
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/]")
            finally:
                rate_limit_sem.release()
                progress.update(task, advance=1)

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_one, fig) for fig in pending]
            for _ in as_completed(futures):
                pass

    # Summary
    console.print(f"\n[bold]Results:[/] {len(descriptions)} described, {errors} errors")

    # Merge current run descriptions with previously processed state
    # CRITICAL: must include ALL processed figures (from this run + previous runs)
    all_descriptions = {
        **{k: v.get("content", "") if isinstance(v, dict) else v
           for k, v in state.get("processed", {}).items()
           if k not in descriptions},  # previously processed (from state)
        **descriptions,  # current run results take precedence
    }

    output_path = book_dir / f"{book_name}_enhanced.md"
    inserted = assemble_enhanced_markdown(md_path, figures, all_descriptions, output_path)
    console.print(f"[green]✅ Assembled {output_path} ({inserted} VLM descriptions inserted)[/]")


if __name__ == "__main__":
    main()

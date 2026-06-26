#!/usr/bin/env python3
"""
vlm_describe.py — Describe figures via self-hosted Ollama VLM.

Phase 3 of PDF Ingestion Pipeline (runs on Vast.ai GPU instance):
  Takes prompt_config.json + figure JPEGs, processes each figure
  via Ollama (Qwen2.5-VL-7B-Instruct), saves results, assembles <book>_enhanced.md.

Resumable — state saved to .vlm_job/ in the staging directory.

Usage:
    python3 /workspace/vlm_describe.py /workspace/staging/<bookname>/
    python3 /workspace/vlm_describe.py /workspace/staging/<bookname>/ --max-workers 4
    python3 /workspace/vlm_describe.py /workspace/staging/<bookname>/ --status

API: ollama Python client — ollama.chat(model, messages=[{images: […]}])
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
        prompt: VLM prompt for this category from prompt_config.json
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


# ── Figure Discovery ──────────────────────────────────────────────────────────
def discover_figures(book_dir: Path) -> list[dict]:
    """
    Find all figure JPEGs and their captions from markdown.
    Returns list of {name, path, caption, category}.
    """
    # Find markdown
    md_files = list(book_dir.glob("*.md"))
    if not md_files:
        return []
    md_path = md_files[0]

    # Parse figure references from markdown
    with open(md_path, encoding="utf-8") as f:
        md_content = f.read()

    # Find all figure references: ![...](_page_XXX_...jpeg)
    # marker-pdf format: ![](_page_3_Figure_8.jpeg) or ![](_page_12_Picture_2.jpeg)
    figure_refs = re.findall(r'!\[(.*?)\]\((_page_\d+_[^.]+\.[jJ][pP][eE]?[gG])\)', md_content)

    # Find all JPEGs on disk
    jpegs = {p.name: p for p in book_dir.glob("_page_*.jp*g")}

    figures = []
    for caption, filename in figure_refs:
        if filename in jpegs:
            figures.append({
                "name": filename.replace(".jpeg", "").replace(".jpg", ""),
                "path": str(jpegs[filename]),
                "caption": caption.strip(),
                "category": "unknown"  # Will be assigned later
            })

    return figures


def classify_figures(figures: list[dict], prompt_config: dict) -> list[dict]:
    """Assign category to each figure based on caption keywords."""
    category_keywords = {
        "circuit": ["circuit", "schematic", "amplifier", "oscillator", "filter",
                     "transistor", "diode", "op-amp", "op amp", "voltage",
                     "current", "signal", "bias", "bjt", "mosfet", "fet"],
        "graph": ["plot", "graph", "curve", "response", "characteristic",
                   "frequency", "gain", "phase", "bode", "impedance",
                   "transfer function", "v/i", "i-v"],
        "block_diagram": ["block diagram", "block", "system", "architecture",
                          "flow", "pipeline", "stage", "module"],
        "photo": ["photo", "photograph", "apparatus", "setup", "equipment",
                   "oscilloscope", "lab", "breadboard", "pcb", "board"]
    }

    for fig in figures:
        caption_lower = (fig["caption"] + " " + fig["name"]).lower()
        best_cat = "graph"  # default
        best_score = 0
        for cat, keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in caption_lower)
            if score > best_score:
                best_score = score
                best_cat = cat
        fig["category"] = best_cat

    return figures


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

    # Check prompt_config.json exists
    pconfig_path = book_dir / "prompt_config.json"
    if not pconfig_path.exists():
        console.print("[red]ERROR:[/] prompt_config.json not found. Run vlm_prompt_gen.py first.")
        sys.exit(1)

    with open(pconfig_path) as f:
        prompt_config = json.load(f)

    # Setup job directory
    job_dir = book_dir / ".vlm_job"
    job_dir.mkdir(exist_ok=True)

    # Discover figures
    figures = discover_figures(book_dir)
    figures = classify_figures(figures, prompt_config)
    console.print(f"[VLM] Found {len(figures)} figures in {book_name}")

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
        cat = fig["category"]
        if cat not in prompt_config:
            console.print(f"[yellow]WARN:[/] No prompt for category '{cat}', skipping {fig['name']}")
            continue
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
    errors_count = 0
    rate_limit_sem = threading.Semaphore(args.max_workers)

    if pending:
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
                nonlocal errors_count
                rate_limit_sem.acquire()
                try:
                    cat = fig["category"]
                    prompt = prompt_config[cat]["prompt"]
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
                            "content": desc
                        }
                        save_state(job_dir, state)
                except Exception as e:
                    with lock:
                        errors_count += 1
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

        console.print(f"\n[bold]Results:[/] {len(descriptions)} described, {errors_count} errors")

    # Merge current run descriptions with previously processed state
    all_descriptions = {
        **{k: v.get("content", "") if isinstance(v, dict) else v
           for k, v in state.get("processed", {}).items()
           if k not in descriptions},
        **descriptions,
    }

    output_path = book_dir / f"{book_name}_enhanced.md"
    inserted = assemble_enhanced_markdown(md_path, figures, all_descriptions, output_path)
    console.print(f"[green]✅ Assembled {output_path} ({inserted} VLM descriptions inserted)[/]")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
vlm_describe.py — VLM Figure Description via Ollama (Qwen2.5-VL-7B).
Phase C of PDF Ingestion Pipeline.
Reads figure_metadata.json from Phase B, describes each figure via Ollama VLM,
and assembles enhanced markdown with VLM descriptions.

Usage:
  python3 /workspace/vlm_describe.py /workspace/staging/<bookname>/
  python3 /workspace/vlm_describe.py /workspace/staging/<bookname>/ --sample 5

Ref: Context7 Ollama docs — chat with images, temperature, num_predict
"""

import os
import sys
import json
import base64
import argparse
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, TaskProgressColumn, TimeRemainingColumn
)

console = Console()

OLLAMA_MODEL = "qwen2.5vl:7b"
DEFAULT_MAX_WORKERS = 2


# ── State Management ──────────────────────────────────────────────────────────

def load_state(job_dir: Path) -> dict:
    """Load job state from .vlm_job/state.json."""
    state_file = job_dir / "state.json"
    if state_file.exists():
        with open(state_file, encoding="utf-8") as f:
            return json.load(f)
    return {"processed": {}, "started_at": None}


def save_state(job_dir: Path, state: dict):
    """Save job state."""
    state_file = job_dir / "state.json"
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def save_error(job_dir: Path, figure_name: str, error_msg: str):
    """Record a per-figure error."""
    err_file = job_dir / "errors.json"
    errors = {}
    if err_file.exists():
        with open(err_file, encoding="utf-8") as f:
            errors = json.load(f)
    errors[figure_name] = {"error": error_msg, "timestamp": str(datetime.now())}
    with open(err_file, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)


def load_errors(job_dir: Path) -> dict:
    """Load error log."""
    err_file = job_dir / "errors.json"
    if err_file.exists():
        with open(err_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ── Figure Metadata (from Phase B) ────────────────────────────────────────────

def load_figure_metadata(book_dir: Path) -> dict:
    """Load figure_metadata.json — the data contract from Phase B."""
    metadata_path = book_dir / "figure_metadata.json"
    if not metadata_path.exists():
        console.print(f"[red]ERROR:[/] {metadata_path} not found")
        console.print("Run vlm_prompt_gen.py (Phase B) first to generate figure metadata.")
        sys.exit(1)

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    # Map JPEG files to figure entries
    jpeg_map = {}
    for jpeg_path in book_dir.glob("_page_*.jp*g"):
        filename = jpeg_path.name
        jpeg_map[filename] = str(jpeg_path)

    # Also check nested dir
    nested_dir = None
    for d in book_dir.iterdir():
        if d.is_dir() and any(d.glob("_page_*.jp*g")):
            nested_dir = d
            break
    if nested_dir:
        for jpeg_path in nested_dir.glob("_page_*.jp*g"):
            if jpeg_path.name not in jpeg_map:
                jpeg_map[jpeg_path.name] = str(jpeg_path)

    for fig in metadata.get("figures", []):
        filename = fig.get("filename", "")
        if filename in jpeg_map:
            fig["path"] = jpeg_map[filename]
        else:
            fig["path"] = str(book_dir / filename)

    return metadata


# ── VLM Description ───────────────────────────────────────────────────────────

def describe_figure(image_path: Path, category: str, prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call Ollama VLM to describe a single figure."""
    from ollama import chat

    if not image_path or not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    full_prompt = f"{prompt}\n\nFigure category: {category}"

    response = chat(
        model=model,
        messages=[{
            "role": "user",
            "content": full_prompt,
            "images": [img_b64]
        }],
        options={
            "temperature": 0.3,
            "num_predict": 1024
        }
    )

    return response["message"]["content"].strip()


# ── Enhanced Markdown Assembly ─────────────────────────────────────────────────

def assemble_enhanced_markdown(md_path: Path, figures: list[dict],
                               descriptions: dict[str, str],
                               output_path: Path) -> int:
    """Insert VLM descriptions into markdown after each figure reference."""
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Build lookup: _page_31_Figure_6 → name without extension
    name_to_desc = {}
    for desc_name, desc_text in descriptions.items():
        # desc_name might be "_page_31_Figure_6" (from discover) or full path
        clean = desc_name.split("/")[-1].replace(".jpeg", "").replace(".jpg", "")
        name_to_desc[clean] = desc_text

    # Also index by filename
    for fig in figures:
        name = fig.get("name", "")
        if name in name_to_desc:
            key = name
        else:
            key = fig.get("filename", "").replace(".jpeg", "").replace(".jpg", "")
        if key not in name_to_desc:
            continue

    new_lines = []
    inserted = 0

    for line in lines:
        new_lines.append(line)
        if line.startswith("![") and "(_page_" in line:
            match = __import__("re").match(
                r'!\[(.*?)\]\((_page_\d+_[^.]+.[jJ][pP][eE]?[gG])\)', line.strip()
            )
            if match:
                filename = match.group(2)
                name = filename.replace(".jpeg", "").replace(".jpg", "")
                name_no_ext = __import__("re").sub(r'\.[jJ][pP][eE]?[gG]$', '', filename)
                cat = "unknown"
                for fig in figures:
                    if fig.get("filename") == filename or fig.get("name") == name_no_ext:
                        cat = fig.get("category", "unknown")
                        break
                desc = name_to_desc.get(name, name_to_desc.get(name_no_ext, ""))
                if desc:
                    new_lines.append(f"\n<details>\n<summary><b>🤖 VLM Description</b> ({cat})</summary>\n\n")
                    new_lines.append(desc)
                    new_lines.append("\n</details>\n")
                    inserted += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

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

    md_files = list(book_dir.glob("*.md"))
    if not md_files:
        console.print(f"[red]ERROR:[/] No .md file found in {book_dir}")
        sys.exit(1)
    md_path = md_files[0]
    book_name = md_path.stem

    job_dir = book_dir / ".vlm_job"
    job_dir.mkdir(exist_ok=True)

    metadata = load_figure_metadata(book_dir)
    figures = metadata["figures"]
    category_prompts = metadata.get("category_prompts", {})

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

    if args.sample > 0:
        figures = figures[:args.sample]
        console.print(f"[VLM] Sample mode: {len(figures)} figures")

    state = load_state(job_dir)
    if not state.get("started_at"):
        state["started_at"] = str(datetime.now())
    save_state(job_dir, state)

    pending = []
    for fig in figures:
        if not args.force and fig["name"] in state.get("processed", {}):
            continue
        cat = fig.get("category", "other_unknown")
        prompt = fig.get("prompt", "")
        if not prompt:
            prompt = category_prompts.get(cat, {}).get("prompt", "")
        if not prompt:
            console.print(f"[yellow]WARN:[/] No prompt for '{fig['name']}' (category: {cat}), skipping")
            continue
        fig["_prompt"] = prompt
        pending.append(fig)

    if not pending:
        console.print(f"[green]✅ All {len(figures)} figures already processed.[/]")
    else:
        console.print(f"[VLM] Processing {len(pending)} pending figures "
                      f"(model: {args.model}, workers: {args.max_workers})")

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
                        "content": desc
                    }
                    save_state(job_dir, state)
                    done = len(state.get("processed", {}))
                    print(f"[VLM] {done}/{len(figures)} {fig['name']} ({cat})", flush=True)
            except Exception as e:
                with lock:
                    errors += 1
                    save_error(job_dir, fig["name"], str(e))
                    console.print(f"[red]ERROR {fig['name']}:[/] {e}")
            finally:
                rate_limit_sem.release()
                progress.update(task, advance=1)

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_one, fig) for fig in pending]
            for _ in as_completed(futures):
                pass

    console.print(f"\n[bold]Results:[/] {len(descriptions)} described, {errors} errors")

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

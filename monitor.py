#!/usr/bin/env python3
"""
monitor.py — Rich Live monitoring console for Vast.ai GPU pipeline.

Streams SSH logs from the GPU instance in real-time, displays:
  - Pipeline phase progress (marker-pdf → prompt gen → VLM)
  - GPU utilization (nvidia-smi polling)
  - Error detection and alerts
  - Elapsed time, cost estimate

Usage:
  python3 monitor.py <INSTANCE_ID>
  python3 monitor.py <INSTANCE_ID> --refresh 5

Ref: Context7 Rich docs — Live display, Progress bars, Panel layout
"""

import sys
import time
import subprocess
import argparse
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn

console = Console()

# ── SSH ───────────────────────────────────────────────────────────────────────

def parse_ssh_url(url: str) -> tuple[str, int]:
    """Parse vastai ssh-url output (dual format)."""
    if url.startswith("ssh://"):
        url = url.replace("ssh://", "")
        if "@" in url:
            url = url.split("@", 1)[1]
        host, port = url.rsplit(":", 1)
    else:
        parts = url.replace("ssh ", "").split()
        host = parts[0].split("@")[1] if "@" in parts[0] else parts[0]
        port = parts[2] if len(parts) > 2 else "22"
    return host, int(port)


def get_ssh_info(instance_id: str) -> tuple[str, int]:
    """Get SSH host:port from vastai CLI."""
    result = subprocess.run(
        ["vastai", "ssh-url", instance_id],
        capture_output=True, text=True, timeout=15
    )
    return parse_ssh_url(result.stdout.strip())


# ── Log Streaming ─────────────────────────────────────────────────────────────

class LogStreamer:
    """Stream logs from remote instance via SSH tail -f."""

    def __init__(self, host: str, port: int, log_dir: str = "/workspace/output/logs"):
        self.host = host
        self.port = port
        self.log_dir = log_dir
        self.lines: list[str] = []
        self.errors: list[str] = []
        self.gpu_lines: list[str] = []
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._stream, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _stream(self):
        import paramiko
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.host, port=self.port, username="root", timeout=10,
                      banner_timeout=10)

        cmd = f"tail -f /workspace/output/DONE 2>/dev/null & tail -n 200 {self.log_dir}/*.log 2>/dev/null; wait"
        stdin, stdout, stderr = client.exec_command(cmd)
        stdin.close()

        for line in iter(stdout.readline, ""):
            if not self._running:
                break
            line = line.strip()
            if not line:
                continue
            self.lines.append(line)
            if "ERROR" in line or "FAIL" in line or "error" in line:
                self.errors.append(line)

        client.close()


# ── GPU Stats ─────────────────────────────────────────────────────────────────

def get_gpu_stats(host: str, port: int) -> dict:
    """Poll nvidia-smi on remote instance."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
             "-p", str(port), f"root@{host}",
             "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 5:
                return {
                    "name": parts[0],
                    "mem_used": parts[1],
                    "mem_total": parts[2],
                    "util": parts[3],
                    "temp": parts[4],
                }
    except Exception:
        pass
    return {}


# ── Phase Detection ───────────────────────────────────────────────────────────

def detect_phase(log_lines: list[str]) -> str:
    """Detect current pipeline phase from log output."""
    recent = "\n".join(log_lines[-50:])
    if "PHASE C" in recent or "Describing figures" in recent:
        return "PHASE C: VLM Description"
    if "PHASE B" in recent or "Generating prompts" in recent:
        return "PHASE B: DeepSeek Prompt Gen"
    if "PHASE A" in recent or "marker-pdf" in recent or "Extracting:" in recent:
        return "PHASE A: marker-pdf Extraction"
    return "Waiting..."


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Monitor Vast.ai GPU pipeline")
    parser.add_argument("instance_id", help="Vast.ai instance ID")
    parser.add_argument("--refresh", type=int, default=3, help="GPU stat refresh interval (seconds)")
    args = parser.parse_args()

    # Get SSH info
    console.print(f"[cyan]Connecting to instance {args.instance_id}...[/]")
    host, port = get_ssh_info(args.instance_id)
    console.print(f"[green]SSH:[/] root@{host}:{port}")

    # Start log streaming
    streamer = LogStreamer(host, port)
    streamer.start()
    time.sleep(2)  # Let first lines arrive

    start_time = datetime.now()
    gpu_last_poll = 0
    gpu_cache = {}
    cost_per_hour = 0.13  # Default RTX 3090 estimate

    def make_layout() -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="main", ratio=2),
            Layout(name="sidebar", ratio=1),
        )
        return layout

    def render():
        nonlocal gpu_last_poll, gpu_cache

        elapsed = datetime.now() - start_time
        phase = detect_phase(streamer.lines)

        # GPU stats (throttled)
        if time.time() - gpu_last_poll > args.refresh:
            gpu_cache = get_gpu_stats(host, port)
            gpu_last_poll = time.time()

        # Header
        header = Panel(
            f"[bold]PDF Ingestion Pipeline Monitor[/]\n"
            f"Instance: [cyan]{args.instance_id}[/] | "
            f"Elapsed: [yellow]{str(elapsed).split('.')[0]}[/] | "
            f"Phase: [bold cyan]{phase}[/]",
            style="cyan"
        )

        # Main: recent logs
        recent = streamer.lines[-30:]
        log_text = "\n".join(recent) if recent else "[dim]Waiting for logs...[/]"

        # Highlight errors
        for err in streamer.errors[-5:]:
            log_text = log_text.replace(err, f"[red]{err}[/]")

        main_panel = Panel(log_text, title="Pipeline Log", border_style="blue")

        # Sidebar: GPU + progress
        sidebar_parts = []
        if gpu_cache:
            sidebar_parts.append(
                Panel(
                    f"GPU: {gpu_cache.get('name', '?')}\n"
                    f"VRAM: {gpu_cache.get('mem_used', '?')} / {gpu_cache.get('mem_total', '?')}\n"
                    f"Util: {gpu_cache.get('util', '?')}\n"
                    f"Temp: {gpu_cache.get('temp', '?')}",
                    title="GPU Stats", border_style="green"
                )
            )
        else:
            sidebar_parts.append(Panel("[dim]No GPU data[/]", title="GPU Stats", border_style="green"))

        # Cost estimate
        hours = elapsed.total_seconds() / 3600
        cost = hours * cost_per_hour
        sidebar_parts.append(
            Panel(
                f"Rate: ${cost_per_hour:.2f}/hr\n"
                f"Elapsed: {elapsed}\n"
                f"Est. cost: [yellow]${cost:.4f}[/]",
                title="Cost", border_style="yellow"
            )
        )

        # Errors
        if streamer.errors:
            sidebar_parts.append(
                Panel(
                    f"{len(streamer.errors)} errors\n" +
                    "\n".join(streamer.errors[-5:]),
                    title="⚠️ Errors", border_style="red"
                )
            )

        sidebar = Panel("\n".join(str(p) for p in sidebar_parts) if sidebar_parts else "",
                       title="Status", border_style="magenta")

        layout = make_layout()
        layout["header"].update(header)
        layout["main"].update(main_panel)
        layout["sidebar"].update(sidebar)

        # Footer
        log_count = len(streamer.lines)
        last_line = streamer.lines[-1][:80] + "..." if streamer.lines else "—"
        footer = Panel(
            f"Log lines: {log_count} | Latest: [dim]{last_line}[/] | Ctrl+C to exit",
            style="dim"
        )
        layout["footer"].update(footer)

        return layout

    try:
        with Live(render(), refresh_per_second=2, console=console) as live:
            while True:
                time.sleep(0.5)
                live.update(render())
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped.[/]")
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()
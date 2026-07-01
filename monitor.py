#!/usr/bin/env python3
"""
monitor.py — Rich Live monitoring console for Vast.ai GPU pipeline.
"""
import time
import os
import subprocess
import argparse
import threading
from datetime import datetime
from typing import Optional
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
console = Console()
def parse_ssh_url(url: str) -> tuple[str, int]:
    if url.startswith("ssh://"):
        url = url.replace("ssh://", "")
        if "@" in url: url = url.split("@", 1)[1]
        host, port = url.rsplit(":", 1)
    else:
        parts = url.replace("ssh ", "").split()
        host = parts[0].split("@")[1] if "@" in parts[0] else parts[0]
        port = parts[2] if len(parts) > 2 else "22"
    return host, int(port)
def get_ssh_info(instance_id: str) -> tuple[str, int]:
    result = subprocess.run(["vastai", "ssh-url", instance_id], capture_output=True, text=True, timeout=15)
    return parse_ssh_url(result.stdout.strip())
def _resolve_key() -> str:
    from pathlib import Path as _Path
    env_key = os.environ.get("MONITOR_SSH_KEY", "")
    if env_key and _Path(env_key).exists(): return env_key
    hermes_key = _Path("/home/hermeswebui/.hermes/home/.ssh/id_ed25519")
    if hermes_key.exists(): return str(hermes_key)
    host_key = _Path.home() / ".ssh" / "id_ed25519"
    if host_key.exists(): return str(host_key)
    return str(host_key)
JUMP_HOST = os.environ.get("MONITOR_JUMP_HOST", "100.72.250.8")
JUMP_USER = os.environ.get("MONITOR_JUMP_USER", "nakalab")
JUMP_KEY = _resolve_key()
TARGET_KEY = JUMP_KEY
def _is_jump_host() -> bool:
    import socket
    try: return socket.gethostname().startswith("alpha") or os.uname().nodename.startswith("alpha")
    except Exception: return False
NEED_PROXY = not _is_jump_host()
class LogStreamer:
    def __init__(self, host: str, port: int, log_dir: str = "/workspace/staging", ssh_key: str | None = None):
        self.host = host; self.port = port; self.log_dir = log_dir
        self.ssh_key = ssh_key or TARGET_KEY
        self.lines: list[str] = []; self.errors: list[str] = []
        self._thread: Optional[threading.Thread] = None; self._running = False
    def start(self): self._running = True; self._thread = threading.Thread(target=self._stream, daemon=True); self._thread.start()
    def stop(self): self._running = False
    def _stream(self):
        import paramiko
        client = paramiko.SSHClient(); client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        pkey = paramiko.Ed25519Key.from_private_key_file(self.ssh_key); extra_kwargs = {}
        if NEED_PROXY:
            proxy_jump = f"ssh -i {JUMP_KEY} -o StrictHostKeyChecking=no -W {self.host}:{self.port} {JUMP_USER}@{JUMP_HOST}"
            extra_kwargs["sock"] = paramiko.ProxyCommand(proxy_jump)
        client.connect(self.host, port=self.port, username="root", timeout=10, banner_timeout=10, pkey=pkey, **extra_kwargs)
        cmd = f"find {self.log_dir} -name '*.log' 2>/dev/null | while read f; do tail -f \"$f\" 2>/dev/null & done; tail -f /workspace/output/DONE 2>/dev/null & wait"
        stdin, stdout, stderr = client.exec_command(cmd); stdin.close()
        for line in iter(stdout.readline, ""):
            if not self._running: break
            line = line.strip()
            if line: self.lines.append(line)
            if "ERROR" in line or "FAIL" in line or "error" in line: self.errors.append(line)
        client.close()
def get_gpu_stats(host: str, port: int, ssh_key: str | None = None) -> dict:
    key = ssh_key or TARGET_KEY
    try:
        ssh_args = ["ssh","-i",key,"-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=/dev/null","-o","ConnectTimeout=5"]
        if NEED_PROXY:
            pc = f"ssh -i {JUMP_KEY} -o StrictHostKeyChecking=no -W {host}:{port} {JUMP_USER}@{JUMP_HOST}"
            ssh_args.extend(["-o",f"ProxyCommand={pc}"])
        ssh_args.extend(["-p",str(port),f"root@{host}","nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null"])
        result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 5: return {"name":parts[0],"mem_used":parts[1],"mem_total":parts[2],"util":parts[3],"temp":parts[4]}
    except Exception: pass
    return {}
def detect_phase(log_lines: list[str]) -> str:
    recent = "\n".join(log_lines[-50:])
    if "[VLM]" in recent or "Describing" in recent or "PHASE C" in recent: return "PHASE C: VLM Description"
    if "[METADATA]" in recent or "PHASE B" in recent or "Generating prompts" in recent: return "PHASE B: DeepSeek Prompt Gen"
    if "PHASE A" in recent or "marker-pdf" in recent or "Recognizing" in recent: return "PHASE A: marker-pdf Extraction"
    return "Idle / Waiting..."
def main():
    parser = argparse.ArgumentParser(description="Monitor Vast.ai GPU pipeline")
    parser.add_argument("instance_id", help="Vast.ai instance ID")
    parser.add_argument("--refresh", type=int, default=3, help="GPU stat refresh interval (seconds)")
    parser.add_argument("--log-dir", default="/workspace/staging", help="Directory for log files (default: /workspace/staging)")
    parser.add_argument("--ssh-key", default=None, help="Path to SSH private key (default: auto-detect)")
    args = parser.parse_args()
    ssh_key = args.ssh_key or TARGET_KEY
    console.print(f"[cyan]Connecting to instance {args.instance_id}...[/]")
    host, port = get_ssh_info(args.instance_id)
    console.print(f"[green]SSH:[/] root@{host}:{port} (key: {ssh_key})")
    streamer = LogStreamer(host, port, log_dir=args.log_dir, ssh_key=ssh_key)
    streamer.start(); time.sleep(2)
    start_time = datetime.now(); gpu_last_poll = 0; gpu_cache = {}; cost_per_hour = 0.13
    def make_layout() -> Layout:
        layout = Layout(); layout.split_column(Layout(name="header", size=3), Layout(name="body"), Layout(name="footer", size=3))
        layout["body"].split_row(Layout(name="main", ratio=2), Layout(name="sidebar", ratio=1))
        return layout
    def render():
        nonlocal gpu_last_poll, gpu_cache
        elapsed = datetime.now() - start_time; phase = detect_phase(streamer.lines)
        if time.time() - gpu_last_poll > args.refresh: gpu_cache = get_gpu_stats(host, port, ssh_key=ssh_key); gpu_last_poll = time.time()
        header = Panel(f"[bold]PDF Ingestion Pipeline Monitor[/]\nInstance: [cyan]{args.instance_id}[/] | Elapsed: [yellow]{str(elapsed).split('.')[0]}[/] | Phase: [bold cyan]{phase}[/]", style="cyan")
        recent = streamer.lines[-30:]; log_text = "\n".join(recent) if recent else "[dim]Waiting for logs...[/]"
        for err in streamer.errors[-5:]: log_text = log_text.replace(err, f"[red]{err}[/]")
        main_panel = Panel(log_text, title="Pipeline Log", border_style="blue")
        sidebar_parts = []
        if gpu_cache: sidebar_parts.append(Panel(f"GPU: {gpu_cache.get('name','?')}\nVRAM: {gpu_cache.get('mem_used','?')} / {gpu_cache.get('mem_total','?')}\nUtil: {gpu_cache.get('util','?')}\nTemp: {gpu_cache.get('temp','?')}", title="GPU Stats", border_style="green"))
        else: sidebar_parts.append(Panel("[dim]No GPU data[/]", title="GPU Stats", border_style="green"))
        hours = elapsed.total_seconds() / 3600; cost = hours * cost_per_hour
        sidebar_parts.append(Panel(f"Rate: ${cost_per_hour:.2f}/hr\nElapsed: {elapsed}\nEst. cost: [yellow]${cost:.4f}[/]", title="Cost", border_style="yellow"))
        if streamer.errors: sidebar_parts.append(Panel(f"{len(streamer.errors)} errors\n"+"\n".join(streamer.errors[-5:]), title="⚠️ Errors", border_style="red"))
        sidebar = Group(*sidebar_parts) if sidebar_parts else Group()
        layout = make_layout(); layout["header"].update(header); layout["main"].update(main_panel); layout["sidebar"].update(sidebar)
        log_count = len(streamer.lines); last_line = streamer.lines[-1][:80]+"..." if streamer.lines else "—"
        layout["footer"].update(Panel(f"Log lines: {log_count} | Latest: [dim]{last_line}[/] | Ctrl+C to exit", style="dim"))
        return layout
    try:
        with Live(render(), refresh_per_second=2, console=console) as live:
            while True: time.sleep(0.5); live.update(render())
    except KeyboardInterrupt: console.print("\n[yellow]Monitoring stopped.[/]")
    finally: streamer.stop()
if __name__ == "__main__": main()

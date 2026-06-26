#!/usr/bin/env python3
"""
orchestrate.py — Agent-side Vast.ai GPU Instance Orchestrator.

Manages the full lifecycle of a PDF Ingestion Pipeline job:
  0. Create reusable Vast.ai template (once)
  1. Search cheapest GPU → user creates instance → hand Instance ID to agent
  2. Upload PDFs via SCP
  3. Upload .env.deepseek → /workspace/.env on instance
  4. SSH exec run_pipeline.sh
  5. Monitor progress via SSH log streaming
  6. Download results to local
  7. Report done (no destroy — user decides)

Environment:
  VAST_API_KEY — from .env.vastai (agent-side, NEVER uploaded to instance)
  DEEPSEEK_API_KEY — from .env.deepseek (agent-side, SCP'd to instance)

Usage:
  python3 orchestrate.py template        # Create reusable template
  python3 orchestrate.py instance <ID>   # Manage existing instance

Ref: gpu-cloud-rental skill — 17 operational pitfalls documented.
"""

import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# ── Rich ──────────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel

console = Console()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ENV_VASTAI = BASE_DIR / ".env.vastai"
ENV_DEEPSEEK = BASE_DIR / ".env.deepseek"
TEMPLATE_CONFIG = BASE_DIR / "template_config.json"
INPUT_DIR = Path("/workspace_alpha/PDF_Ingestion_Pipeline/pdf_input")
OUTPUT_DIR = Path("/workspace_alpha/PDF_Ingestion_Pipeline/resultv2")

VAST_API_BASE = "https://console.vast.ai/api"


# ═══════════════════════════════════════════════════════════════════════════════
# API Key Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_env_file(path: Path, key: str) -> str:
    """Load a single KEY=VALUE from an env file."""
    if not path.exists():
        console.print(f"[red]ERROR:[/] {path} not found")
        sys.exit(1)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    console.print(f"[red]ERROR:[/] {key} not found in {path}")
    sys.exit(1)


def get_vast_api_key() -> str:
    return load_env_file(ENV_VASTAI, "VAST_API_KEY")


def get_deepseek_api_key() -> str:
    return load_env_file(ENV_DEEPSEEK, "DEEPSEEK_API_KEY")


# ═══════════════════════════════════════════════════════════════════════════════
# Vast.ai API Helpers
# ═══════════════════════════════════════════════════════════════════════════════

import urllib.request


def vast_api(method: str, path: str, data: dict = None, api_key: str = None) -> dict:
    """Call Vast.ai REST API."""
    if api_key is None:
        api_key = get_vast_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        f"{VAST_API_BASE}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


# ═══════════════════════════════════════════════════════════════════════════════
# Template Management
# ═══════════════════════════════════════════════════════════════════════════════

def create_template() -> str:
    """Create a reusable Vast.ai template. Returns template hash ID."""
    with open(TEMPLATE_CONFIG) as f:
        config = json.load(f)

    console.print(f"[cyan]Creating template: {config['name']}...[/]")
    result = vast_api("POST", "/v0/template/", config)
    template = result.get("template", {})
    hash_id = template.get("hash_id", "")
    console.print(f"[green]✅ Template created:[/] {hash_id}")
    console.print(f"    Image: {config['image']}:{config['tag']}")
    console.print(f"    Disk: {config['recommended_disk_space']}GB, SSH direct, Private")
    return hash_id


# ═══════════════════════════════════════════════════════════════════════════════
# GPU Search
# ═══════════════════════════════════════════════════════════════════════════════

def search_gpus(max_price: float = 0.30) -> list[dict]:
    """Search cheapest RTX 3090/4090, verified first, then unverified fallback."""
    console.print("[cyan]Searching GPU offers...[/]")

    for verified in [True, False]:
        ver_str = "verified=true" if verified else ""
        tag = "verified" if verified else "unverified"

        cmd = f"vastai search offers 'gpu_name=RTX_3090 gpu_name=RTX_4090 num_gpus=1 rentable=true gpu_ram>=20000 direct_port_count>=1 inet_down>=500 {ver_str}' -o dph_total+ --raw --limit 5"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            continue

        try:
            offers = json.loads(result.stdout)
        except json.JSONDecodeError:
            continue

        if offers:
            console.print(f"[green]Found {len(offers)} {tag} offers[/]")
            for o in offers[:3]:
                console.print(f"  {o.get('id', '?'):>10}  {o.get('gpu_name', '?'):<15}  "
                            f"${o.get('dph_total', 0):.4f}/hr  {o.get('geolocation', '?')}")
            return offers[:3]

    console.print("[red]No GPU offers found[/]")
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# SSH Helpers
# ═══════════════════════════════════════════════════════════════════════════════

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


def get_instance_ssh(instance_id: str) -> tuple[str, int]:
    """Get SSH host:port for a running instance."""
    result = subprocess.run(
        ["vastai", "ssh-url", instance_id],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode != 0:
        raise RuntimeError(f"vastai ssh-url failed: {result.stderr}")

    url = result.stdout.strip()
    return parse_ssh_url(url)


def ssh_exec(host: str, port: int, command: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """Execute command on instance via SSH."""
    return subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
         "-p", str(port), f"root@{host}", command],
        capture_output=True, text=True, timeout=timeout
    )


def scp_upload(host: str, port: int, local_path: str, remote_path: str):
    """Upload a file to instance via SCP pipe (bypasses ~ expansion issues)."""
    console.print(f"[dim]SCP: {local_path} → {remote_path}[/]")
    with open(local_path, "rb") as f:
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-p", str(port), f"root@{host}", f"cat > {remote_path}"],
            input=f.read(), timeout=60, check=True
        )


def scp_download(host: str, port: int, remote_path: str, local_path: str):
    """Download a file from instance via SCP."""
    console.print(f"[dim]SCP: {remote_path} → {local_path}[/]")
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
         "-p", str(port), f"root@{host}", f"cat {remote_path}"],
        capture_output=True, timeout=60
    )
    if result.returncode == 0:
        with open(local_path, "wb") as f:
            f.write(result.stdout)
    else:
        console.print(f"[red]SCP download failed:[/] {result.stderr}")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

def upload_pdfs(host: str, port: int):
    """Upload all pending PDFs to instance input directory."""
    pdfs = list(INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        console.print("[yellow]No PDFs in pdf_input/[/]")
        return

    # Ensure input dir exists
    ssh_exec(host, port, "mkdir -p /workspace/input /workspace/staging /workspace/output")

    for pdf in pdfs:
        scp_upload(host, port, str(pdf), f"/workspace/input/{pdf.name}")
        console.print(f"[green]✅ Uploaded:[/] {pdf.name}")


def upload_env(host: str, port: int):
    """Upload .env.deepseek as /workspace/.env on instance."""
    scp_upload(host, port, str(ENV_DEEPSEEK), "/workspace/.env")
    console.print("[green]✅ Uploaded .env.deepseek → /workspace/.env[/]")


def run_pipeline(host: str, port: int) -> int:
    """Execute run_pipeline.sh on instance."""
    console.print("[cyan]Starting pipeline on instance...[/]")
    result = ssh_exec(host, port, "bash /workspace/run_pipeline.sh 2>&1", timeout=3600)
    console.print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.stderr:
        # Filter Vast.ai MOTD noise
        stderr_lines = [line for line in result.stderr.split("\n")
                       if "Welcome to vast.ai" not in line
                       and "Have fun!" not in line
                       and "vast-agents-guide" not in line]
        if stderr_lines:
            console.print(f"[yellow]Stderr:[/] {''.join(stderr_lines)[:500]}")
    return result.returncode


def download_results(host: str, port: int):
    """Download _enhanced.md files from instance output dir."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = ssh_exec(host, port, "find /workspace/output -name '*_enhanced.md' -type f")
    files = [f.strip() for f in result.stdout.split("\n") if f.strip()]

    if not files:
        console.print("[yellow]No _enhanced.md files found on instance[/]")
        return

    for remote_path in files:
        fname = Path(remote_path).name
        local_path = OUTPUT_DIR / fname
        scp_download(host, port, remote_path, str(local_path))
        console.print(f"[green]✅ Downloaded:[/] {fname} ({local_path.stat().st_size:,} bytes)")

    # Also download logs
    ssh_exec(host, port, "tar czf /workspace/output/logs.tar.gz /workspace/output/logs/ 2>/dev/null; true")
    logs_tar = OUTPUT_DIR / f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    scp_download(host, port, "/workspace/output/logs.tar.gz", str(logs_tar))
    if logs_tar.exists() and logs_tar.stat().st_size > 100:
        console.print(f"[green]✅ Downloaded logs:[/] {logs_tar.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_template():
    """Create reusable Vast.ai template."""
    hash_id = create_template()
    console.print("\n[bold cyan]Template ready.[/] Use this hash when creating instances:")
    console.print(f"  [bold]vastai create instance <OFFER_ID> --template_hash {hash_id} --disk 50[/]")


def cmd_instance(instance_id: str):
    """Manage an existing instance: upload → run → download."""
    console.print(Panel(f"[bold]PDF Ingestion Pipeline — Instance {instance_id}[/]",
                       style="cyan"))

    # Get SSH info
    console.print("[cyan]Getting SSH connection...[/]")
    host, port = get_instance_ssh(instance_id)
    console.print(f"[green]SSH:[/] root@{host}:{port}")

    # Upload
    upload_env(host, port)
    upload_pdfs(host, port)

    # Run
    console.print("\n[bold yellow]Starting pipeline...[/]")
    exit_code = run_pipeline(host, port)

    # Download
    download_results(host, port)

    # Summary
    if exit_code == 0:
        console.print("\n[bold green]✅ Pipeline complete![/]")
    else:
        console.print(f"\n[bold red]❌ Pipeline failed (exit {exit_code})[/]")
        console.print(f"[yellow]Check logs on instance:[/] ssh -p {port} root@{host} 'cat /workspace/output/logs/*.log'")

    console.print(f"\n[dim]Instance {instance_id} is still running. Destroy manually when done:[/]")
    console.print(f"  [dim]echo y | vastai destroy instance {instance_id}[/]")


def main():
    parser = argparse.ArgumentParser(description="PDF Ingestion Pipeline — Vast.ai Orchestrator")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("template", help="Create reusable Vast.ai template")
    p_inst = sub.add_parser("instance", help="Manage instance (upload → run → download)")
    p_inst.add_argument("instance_id", help="Vast.ai instance ID")
    p_inst.add_argument("--skip-upload", action="store_true", help="Skip PDF/env upload (already done)")

    args = parser.parse_args()

    if args.command == "template":
        cmd_template()
    elif args.command == "instance":
        cmd_instance(args.instance_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
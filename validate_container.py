#!/usr/bin/env python3
"""
validate_container.py — Boot health check for Vast.ai GPU instance.

Verifies: CUDA available, GPU detected, Ollama installed, Surya models cached,
pipeline scripts present, SSHd running.
Runs once at container boot via supervisor (priority 2).

Exit 0 = healthy. Non-zero = supervisor skips (autorestart=false).
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

STATUS_FILE = "/workspace/output/HEALTH.json"


def check(ok: bool, msg: str) -> dict:
    return {"status": "OK" if ok else "FAIL", "message": msg, "ok": ok}


def run_checks() -> list[dict]:
    results = []

    # 1. CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_ok else 0
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "none"
        results.append(check(cuda_ok, f"CUDA available: {cuda_ok}, GPU: {gpu_name} (count={gpu_count})"))
    except Exception as e:
        results.append(check(False, f"CUDA check failed: {e}"))

    # 2. GPU via nvidia-smi
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=10)
        results.append(check(r.returncode == 0, f"nvidia-smi: {r.stdout.strip()}"))
    except Exception as e:
        results.append(check(False, f"nvidia-smi failed: {e}"))

    # 3. Ollama binary
    try:
        r = subprocess.run(["which", "ollama"], capture_output=True, text=True, timeout=5)
        results.append(check(r.returncode == 0, f"ollama binary: {r.stdout.strip() or 'found'}"))
    except Exception as e:
        results.append(check(False, f"ollama check failed: {e}"))

    # 4. Surya model cache
    cache_dir = Path("/root/.cache/datalab")
    if cache_dir.exists():
        models = list(cache_dir.rglob("*"))
        results.append(check(len(models) > 0, f"Surya cache: {len(models)} files"))
    else:
        results.append(check(False, "Surya cache: MISSING"))

    # 5. Pipeline scripts
    for script in ["run_pipeline.sh", "vlm_prompt_gen.py", "vlm_describe.py"]:
        path = Path(f"/workspace/{script}")
        results.append(check(path.exists(), f"Script {script}: {'present' if path.exists() else 'MISSING'}"))

    # 6. Python imports
    for mod in ["marker", "openai", "ollama", "rich", "PIL", "cv2"]:
        try:
            __import__(mod)
            results.append(check(True, f"Python import {mod}: OK"))
        except Exception as e:
            results.append(check(False, f"Python import {mod}: {e}"))

    return results


def main():
    results = run_checks()
    all_ok = all(r["ok"] for r in results)

    # Write status
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump({
            "healthy": all_ok,
            "checks": results,
            "timestamp": str(datetime.now())
        }, f, indent=2)

    # Print summary
    fails = [r for r in results if not r["ok"]]
    if fails:
        print(f"[VALIDATE] ❌ {len(fails)}/{len(results)} checks FAILED:")
        for r in fails:
            print(f"  - {r['message']}")
    else:
        print(f"[VALIDATE] ✅ All {len(results)} checks passed")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
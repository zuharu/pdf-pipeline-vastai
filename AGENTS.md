# AGENTS.md — PDF Ingestion Pipeline (Vast.ai GPU)

## Project Identity

- **Name**: `pdf-pipeline-vastai` (Vast.ai GPU deployment)
- **Repo**: `ghcr.io/zuharu/pdf-pipeline-vastai` (GitHub → GHCR)
- **Purpose**: Batch process academic/technical PDFs through marker-pdf GPU extraction → DeepSeek prompt generation → Ollama VLM figure description
- **Workspace**: `/workspace_alpha/PDF_Ingestion_Pipeline/vastai_ingestion_codebase/`
- **Output**: `/workspace_alpha/PDF_Ingestion_Pipeline/resultv2/`

## Architecture (3-Phase, GPU Instance)

```
Phase A: marker-pdf (GPU) → .md + _page_*.jpeg
Phase B: DeepSeek v4-flash (API) → prompt_config.json  
Phase C: Ollama Qwen2.5-VL-7B (GPU) → _enhanced.md
```

Batch optimization: ALL PDFs through Phase A first (Surya loaded once), then ALL through Phase B (no GPU), then ALL through Phase C (Qwen loaded once).

## Key Technical Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Base image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` | Gemini recommended for stability over 2.8.0 |
| VLM | Qwen2.5-VL-7B-Instruct (Q4_K_M) | DocVQA 95.7%, 6.5GB VRAM, Qwen3.5 Ollama bug |
| DeepSeek model | `deepseek-v4-flash` | $0.14/1M input, $0.28/1M output, reasoning_effort=high |
| GPU | RTX 3090 ($0.13/hr) best value | 24GB VRAM sufficient, abundant in Korea/Taiwan |
| Code delivery | Baked into Docker image | Single image, no git clone on instance needed |
| Destroy | User decides (no auto-destroy) | Unlike previous pattern |
| .env | Two files: .env.vastai (agent) + .env.deepseek (instance) | Keeps API keys separate |

## GPU Compatibility Matrix

Our Docker image uses **PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9**. This determines exactly which GPUs work.

### What PyTorch 2.5.1 Supports

```
$ python3 -c "import torch; print(torch.cuda.get_arch_list())"
['sm_50', 'sm_60', 'sm_61', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
```

This maps to NVIDIA GPU generations:

| CC | Generation | Year | Example GPUs |
|----|-----------|------|-------------|
| 5.0, 5.2, 5.3 | Maxwell | 2014 | GTX 980, Tesla M40 |
| 6.0, 6.1 | Pascal | 2016 | GTX 1080, P100 |
| 7.0, 7.2 | Volta | 2017 | V100 |
| 7.5 | Turing | 2018 | RTX 2080, T4 |
| **8.0** | **Ampere (datacenter)** | 2020 | **A100, A40** |
| **8.6** | **Ampere (consumer)** | 2020 | **RTX 3090, RTX 3080, RTX 3070, RTX 3060, A5000, A4000** |
| **8.9** | **Ada Lovelace** | 2022 | **RTX 4090, RTX 4080, RTX 4070, L40S, RTX 6000 Ada** |
| 9.0 | Hopper | 2022 | H100, H800 |
| ❌ 12.0 (sm_120) | **Blackwell** | 2024 | RTX 5090, RTX PRO 4000, RTX PRO 5000 |

### VRAM Budget

```
Phase A (marker-pdf):    peak ~10 GB  (text recognition of 6,128 segments)
Phase C (Ollama VLM):    ~6.5 GB      (Qwen2.5-VL-7B Q4_K_M quantization)
                          ─────────────────
                          Minimum: 12 GB  (tight — 10GB marker peak + OS overhead)
                          Recommended: ≥24 GB
```

Phases don't run simultaneously — peak VRAM is the max of Phases A and C, not the sum.

### NVIDIA Driver Requirement

Any driver ≥ **525.60.13** supports CUDA 12.x. Vast.ai hosts typically run R535-R570 branch (well above minimum).

### ✅ Fully Compatible GPUs (Vast.ai offer candidates)

| GPU | CC | VRAM | Vast.ai $/hr (unverified) | Best For |
|-----|-----|------|--------------------------|----------|
| **RTX 3090** 🏆 | 8.6 | 24 GB | $0.12–0.16 | **Best value** — 24GB, cheap, abundant Asia |
| **RTX 4090** 🚀 | 8.9 | 24 GB | $0.18–0.27 | **Fastest** — 2× marker-pdf speed, 2× VLM tok/s |
| **L40S** | 8.9 | 48 GB | $0.35–0.50 | Datacenter Ada — overkill VRAM |
| **RTX 6000 Ada** | 8.9 | 48 GB | $0.40–0.60 | Pro Ada — overkill |
| **RTX A6000** | 8.6 | 48 GB | $0.35–0.55 | Pro Ampere — overkill |
| **A100** | 8.0 | 40/80 GB | $0.53–1.00 | Overkill — only if 3090/4090 unavailable |
| **H100** | 9.0 | 80 GB | $1.50–3.00 | Massively overkill |
| **RTX A5000** | 8.6 | 24 GB | $0.20–0.30 | Adequate |
| **RTX A4000** | 8.6 | 16 GB | $0.15–0.25 | Workable (tight) |
| **RTX 3080** | 8.6 | 10 GB | $0.10–0.15 | ⚠️ Borderline — 10GB barely fits marker peak |

### ⚠️ Borderline GPUs (12–16 GB, may OOM on large PDFs)

| GPU | CC | VRAM | Notes |
|-----|-----|------|-------|
| **RTX 3060** | 8.6 | 12 GB | Works for Brophy-size PDFs. May OOM on 800+ page PDFs. |
| **RTX 3070** | 8.6 | 8 GB | ❌ Not recommended — 8GB < 10GB marker peak |
| **RTX 4060 Ti** | 8.9 | 8/16 GB | 16GB variant OK, 8GB variant ❌ |
| **RTX 2080 Ti** | 7.5 | 11 GB | ⚠️ Turing CC 7.5 — works but ~30% slower |
| **T4** | 7.5 | 16 GB | ⚠️ Turing — slow, only if budget-constrained |

### ❌ Incompatible GPUs

| GPU | CC | Why |
|-----|-----|-----|
| **RTX 5090** | sm_120 | Blackwell — no sm_120 kernels in PyTorch 2.5.1 + CUDA 12.4 |
| **RTX PRO 4000** | sm_120 | Blackwell — same |
| **RTX PRO 5000** | sm_120 | Blackwell — same |
| **RTX PRO 6000** | sm_120 | Blackwell — same |
| **GTX 10-series** | 6.1 | Pascal — works but 8GB max, too slow for pipeline |
| **GTX 9-series** | 5.2 | Maxwell — <4GB, useless |

### Vast.ai Search Filters (copy-paste ready)

```bash
# Recommended (24GB+, Ampere or Ada)
vastai search offers 'gpu_ram>=20000 compute_cap>=800 rentable=true direct_port_count>=1 inet_down>=500' -o dph_total+ --raw --limit 10

# Budget (12GB+, any CC 7.5+)
vastai search offers 'gpu_ram>=10000 compute_cap>=750 rentable=true direct_port_count>=1 inet_down>=500' -o dph_total+ --raw --limit 10

# Asia-only (lower latency for Japan user)
vastai search offers 'gpu_ram>=20000 compute_cap>=800 rentable=true direct_port_count>=1 inet_down>=500' -o dph_total+ --raw --limit 100 | python3 -c "
import sys, json
data = json.load(sys.stdin)
for o in data:
    geo = o.get('geolocation', '')
    if any(g in geo.lower() for g in ['japan','taiwan','korea','china','singapore','hong kong']):
        print(f\"{o['id']:>10}  {o['gpu_name']:<20}  \${o['dph_total']:.4f}/hr  {geo}\")
"
```

> **Note:** `compute_cap` in Vast.ai search output is raw integer (e.g. `860` = CC 8.6, `890` = CC 8.9). Filter `compute_cap>=800` for Ampere+Ada.

### Why Blackwell Doesn't Work (Technical Detail)

PyTorch 2.5.1 was released before Blackwell GPUs existed. CUDA 12.4 was released before sm_120 support. The error you'd see on a Blackwell GPU:

```
CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.
```

To support Blackwell, you would need **PyTorch ≥2.7.0 + CUDA ≥12.8** (separate `:blackwell` Docker tag — not built yet).

### Instance Disk

| Item | Size |
|------|------|
| PyTorch base image | ~8 GB |
| Surya models (cached in image) | ~5 GB |
| Ollama + qwen2.5vl:7b (pulled at boot) | ~4.5 GB |
| Pipeline scripts | <1 MB |
| Input PDF + staging output | ~1–5 GB per book |
| **Template recommended disk** | **50 GB** |

## Operational Pitfalls (from gpu-cloud-rental skill)

1. **SSH key must exist before instance creation** — injects at container creation time
2. **Ollama GPU verification** — always check nvidia-smi during inference (can silently fall back to CPU)
3. **transformers<5.0 pin** — required for surya/marker-pdf compatibility
4. **marker_single --output_dir is a FLAG** — not positional argument
5. **Docker-in-Docker blocked on Vast.ai** — build locally + push to GHCR
6. **Vast.ai template env vars** — use `env | grep _ >> /etc/environment` in onstart
7. **Offer ID is CLI/API-only** — not visible in WebUI
8. **gpu_ram is in MEGABYTES** — filter `gpu_ram>=20000` not `>=20`
9. **SCP shell variable `~` expansion** — use full path `/home/user/.ssh/...`
10. **Instance recycle changes port** — always re-run `vastai ssh-url`
11. **Stderr MOTD noise** — filter "Welcome to vast.ai", "Have fun!", "vast-agents-guide"
12. **numpy 2.x conflict** — force `numpy>=1.24,<2.0` when pip-installing on conda images

## Development Workflow

1. Edit files in `/vastai_ingestion_codebase/`
2. Build: `docker build -t ghcr.io/zuharu/pdf-pipeline-vastai:latest .`
3. Push: `docker push ghcr.io/zuharu/pdf-pipeline-vastai:latest`
4. Test on Vast.ai: create instance → SCP PDF → run pipeline → download

## Related Documents

- Vault: `sessions/2026-07-08.md` — Full context recovery + Gemini Deep Research
- Vault: `decisions/2026-06-04-pdf-ingestion-pipeline.md` — Original ADR
- Skill: `gpu-cloud-rental` — 17 operational pitfalls, template API
- Skill: `gpu-cloud-rental/references/pdf-pipeline-docker-strategy.md` — Docker image plan
- Skill: `gpu-cloud-rental/references/marker-pdf-gpu-benchmark.md` — 15.6 min benchmark
# PDF Ingestion Pipeline — Vast.ai GPU Deployment

> **marker-pdf → DeepSeek prompt gen → Ollama Qwen2.5-VL figure description**
> 
> Self-contained Docker image for batch processing academic/technical PDFs on Vast.ai GPU instances.

## Architecture

```
┌─────────────────────────────────────────────────┐
│          VAST.AI GPU INSTANCE                    │
│  Docker: pytorch 2.5.1 + CUDA 12.4              │
│                                                  │
│  PHASE A: marker-pdf (GPU, ~16 min/484pg)        │
│    PDF → .md + _page_*.jpeg                      │
│                                                  │
│  PHASE B: DeepSeek v4-flash (API, ~5 sec/book)   │
│    .md → prompt_config.json                      │
│                                                  │
│  PHASE C: Ollama Qwen2.5-VL-7B (GPU, ~20 min)    │
│    JPEGs + prompts → _enhanced.md                │
└──────────────┬──────────────────────────────────┘
               │ SCP download
               ▼
  /workspace_alpha/PDF_Ingestion_Pipeline/resultv2/
```

## Quickstart

### 1. Build & Push Docker Image

```bash
# Build locally
docker build -t ghcr.io/zuharu/pdf-pipeline-vastai:latest .

# Push to GHCR (or use GitHub Actions CI)
docker push ghcr.io/zuharu/pdf-pipeline-vastai:latest
```

### 2. Create Vast.ai Template (once)

```bash
python3 orchestrate.py template
```

### 3. Rent GPU & Run Pipeline

```bash
# Search cheapest GPU
vastai search offers 'gpu_name=RTX_3090 num_gpus=1 rentable=true gpu_ram>=20000 direct_port_count>=1' -o dph_total+ --raw --limit 5

# Create instance from template
vastai create instance <OFFER_ID> --template_hash <TPL_HASH> --disk 50

# Run pipeline
python3 orchestrate.py instance <INSTANCE_ID>
```

### 4. Monitor Progress

```bash
python3 monitor.py <INSTANCE_ID>
```

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage: builder (warm Surya cache) → runner |
| `requirements.txt` | Python deps (marker-pdf, openai, ollama, rich) |
| `supervisord.conf` | Container process management (sshd + validate) |
| `validate_container.py` | Boot health check (CUDA, GPU, Ollama, imports) |
| `run_pipeline.sh` | Entry script: 3-phase sequential per-PDF |
| `vlm_prompt_gen.py` | DeepSeek API → prompt_config.json |
| `vlm_describe.py` | Ollama vision API → figure descriptions |
| `orchestrate.py` | Agent-side: template → upload → exec → download |
| `monitor.py` | Rich Live console: SSH log stream + GPU stats |
| `template_config.json` | Vast.ai template payload |
| `.github/workflows/docker-build.yml` | CI: build & push to GHCR |

## Local Requirements (for orchestrate.py + monitor.py)

These run on your local machine (not the GPU instance):

### Python packages

```bash
pip install rich paramiko
```

| Package | Why | Used By |
|---------|-----|---------|
| `rich>=13.0` | Terminal tables, panels, progress bars, live display | orchestrate.py, monitor.py |
| `paramiko>=3.0` | SSH client for log streaming (monitor.py) | monitor.py |

### System tools

```bash
pip install vastai
```

| Tool | Why | Used By |
|------|-----|---------|
| `vastai` CLI | GPU search, SSH URL parsing, instance management | orchestrate.py, monitor.py |
| `ssh` | SCP file transfer to instance | orchestrate.py |

### Quick install

```bash
pip install rich paramiko vastai
```

> **Note:** `vastai` CLI requires `VAST_API_KEY` set in `.env.vastai` or via `vastai set api-key <KEY>`.

## Environment

Two `.env` files are required:

| File | Key | Used By | Location |
|------|-----|---------|----------|
| `.env.vastai` | `VAST_API_KEY` | Agent (orchestrate.py) | Local only |
| `.env.deepseek` | `DEEPSEEK_API_KEY` | Instance (vlm_prompt_gen.py) | SCP'd to `/workspace/.env` |

## GPU Tier Support

| Tier | GPUs | Status |
|------|------|--------|
| 🟢 Tier 1 | RTX 3090, RTX 4090, L40S, A100, RTX 6000 Ada | Fully supported |
| 🟡 Tier 2 | RTX 5090, RTX PRO 4000/5000 (Blackwell sm_120) | Experimental |

## Cost per Book (Brophy 484pg, 425 figures)

| GPU | Rate | Time | Cost/Book |
|-----|------|------|-----------|
| RTX 3090 🏆 | $0.13/hr | ~36 min | **$0.08** |
| RTX 4090 | $0.27/hr | ~25 min | **$0.11** |
| OpenRouter (no GPU) | $1.28 flat | ~30 min | **$1.28** |

## VLM Model

**Qwen2.5-VL-7B-Instruct** (`qwen2.5vl:7b` via Ollama)
- DocVQA: 95.7% (highest for document figures)
- VRAM: 6.5 GB (Q4_K_M quantization)
- Speed: ~42 tok/s on RTX 3090
- ⚠️ Qwen3.5 excluded — Ollama bug (output goes to "thinking" field)

## References

- [marker-pdf](https://github.com/datalab-to/marker) — GPU PDF extraction
- [Ollama Python](https://github.com/ollama/ollama-python) — Vision API client
- [DeepSeek API](https://api-docs.deepseek.com) — OpenAI-compatible chat completions
- [Vast.ai Docs](https://docs.vast.ai) — GPU rental platform
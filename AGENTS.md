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

## Operational Pitfalls (from gpu-cloud-rental skill)

1. **SSH key must exist before instance creation** — injects at container creation time
2. **Ollama GPU verification** — always check nvidia-smi during inference (can silently fall back to CPU)
3. **transformers<5.0 pin** — required for surya/marker-pdf compatibility
4. **marker_single --output_dir is a FLAG** — not positional argument
5. **Docker-in-Docker blocked on Vast.ai** — build locally + push to GHCR
6. **Vast.ai template env vars** — use `env | grep _ >> /etc/environment` in onstart
7. **Offer ID is CLI/API-only** — not visible in WebUI
8. **gpu_ram is in MEGABYTES** — filter `gpu_ram>=20000` not `>=20`

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
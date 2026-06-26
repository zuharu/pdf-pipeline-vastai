#!/bin/bash
# ── PDF Ingestion Pipeline — Unified Runner for Vast.ai GPU Instance ──────────
#
# Runs ALL pipeline phases for ALL input PDFs:
#   Phase A: marker-pdf GPU extraction (Surya loaded once for all PDFs)
#   Phase B: DeepSeek prompt generation (API call, no GPU)
#   Phase C: Ollama VLM description (Qwen2.5-VL loaded once for all PDFs)
#
# Usage (on instance):
#   bash /workspace/run_pipeline.sh
#
# Environment:
#   DEEPSEEK_API_KEY — from /workspace/.env (SCP'd by agent)
#   Input: /workspace/input/*.pdf
#   Output: /workspace/staging/<bookname>/ → /workspace/output/

set -euo pipefail
set -o pipefail  # detect failures in piped commands (marker_single | tee)

# ── Timestamp helper ──────────────────────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S'; }

INPUT_DIR="/workspace/input"
STAGING_DIR="/workspace/staging"
OUTPUT_DIR="/workspace/output"
LOG_DIR="/workspace/output/logs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

mkdir -p "$STAGING_DIR" "$OUTPUT_DIR" "$LOG_DIR"

# ── Load environment ──────────────────────────────────────────────────────────
if [ -f /workspace/.env ]; then
    set -a
    source /workspace/.env
    set +a
    echo -e "${GREEN}[ENV] Loaded /workspace/.env${NC}"
else
    echo -e "${YELLOW}[ENV] No /workspace/.env found (DeepSeek API key may be missing)${NC}"
fi

# ── System info ───────────────────────────────────────────────────────────────
echo -e "${CYAN}=== System Info ===${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo "PyTorch CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"
echo ""

# ── PHASE A: marker-pdf GPU extraction (all PDFs) ────────────────────────────
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  PHASE A: marker-pdf GPU Extraction (all PDFs)          ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"

PDF_COUNT=$(ls -1 "$INPUT_DIR"/*.pdf 2>/dev/null | wc -l)
if [ "$PDF_COUNT" -eq 0 ]; then
    echo -e "${RED}ERROR: No PDFs found in $INPUT_DIR${NC}"
    exit 1
fi
echo -e "Found ${GREEN}$PDF_COUNT${NC} PDF(s) to process"

for pdf in "$INPUT_DIR"/*.pdf; do
    BOOKNAME=$(basename "$pdf" .pdf)
    STAGING="$STAGING_DIR/$BOOKNAME"
    mkdir -p "$STAGING"

    echo ""
    echo -e "${YELLOW}[$(ts)][A] Extracting: $BOOKNAME${NC}"

    # marker_single: --output_dir is a FLAG (not positional!) — Context7 verified
    set +e  # don't exit on marker failure — we want to continue with other PDFs
    marker_single "$pdf" \
        --output_dir "$STAGING" \
        --force_ocr \
        2>&1 | tee "$LOG_DIR/${BOOKNAME}_marker.log"
    MARKER_EXIT=$?
    set -e

    # Verify output — marker_single creates nested subdir named after input file
    # e.g. --output_dir /staging/Book → actual output at /staging/Book/Book.md
    if [ $MARKER_EXIT -eq 0 ]; then
        # Check if output is in nested subdirectory (marker_single default behavior)
        if [ -f "$STAGING/$BOOKNAME/$BOOKNAME.md" ]; then
            # Move files from nested subdir up to STAGING root
            mv "$STAGING/$BOOKNAME"/* "$STAGING/" 2>/dev/null || true
            rmdir "$STAGING/$BOOKNAME" 2>/dev/null || true
        fi
        # Now check for .md at STAGING root (either original or moved)
        if [ -f "$STAGING/$BOOKNAME.md" ]; then
            JPEG_COUNT=$(ls -1 "$STAGING"/_page_*.jp*g 2>/dev/null | wc -l)
            echo -e "${GREEN}[$(ts)][A] ✅ $BOOKNAME: $(wc -c < "$STAGING/$BOOKNAME.md") bytes, $JPEG_COUNT figures${NC}"
            touch "$STAGING/.phase_a_ok"
        else
            echo -e "${RED}[$(ts)][A] ❌ $BOOKNAME: no .md output found${NC}"
            echo "EXIT=$MARKER_EXIT  TIMESTAMP=$(ts)" > "$STAGING/.phase_a_error"
            continue
        fi
    else
        echo -e "${RED}[$(ts)][A] ❌ $BOOKNAME: marker-pdf failed (exit=$MARKER_EXIT)${NC}"
        echo "EXIT=$MARKER_EXIT  TIMESTAMP=$(ts)" > "$STAGING/.phase_a_error"
    fi
done

echo ""
echo -e "${GREEN}[A] Phase A complete — all PDFs extracted${NC}"

# ── PHASE B: DeepSeek prompt generation (all staging dirs) ────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  PHASE B: DeepSeek Prompt Generation (API calls)        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"

for staging in "$STAGING_DIR"/*/; do
    BOOKNAME=$(basename "$staging")

    # Skip dirs where Phase A failed (no .md or no .phase_a_ok marker)
    if [ ! -f "$staging/.phase_a_ok" ]; then
        echo -e "${RED}[$(ts)][B] ⏭️  $BOOKNAME: skipped (Phase A failed or incomplete)${NC}"
        continue
    fi

    echo ""
    echo -e "${YELLOW}[$(ts)][B] Generating prompts for: $BOOKNAME${NC}"

    set +e
    python3 /workspace/vlm_prompt_gen.py "$staging" 2>&1 | tee "$LOG_DIR/${BOOKNAME}_prompt_gen.log"
    PROMPT_EXIT=$?
    set -e

    if [ $PROMPT_EXIT -eq 0 ] && [ -f "$staging/prompt_config.json" ]; then
        echo -e "${GREEN}[$(ts)][B] ✅ $BOOKNAME: prompt_config.json ready${NC}"
    else
        echo -e "${RED}[$(ts)][B] ❌ $BOOKNAME: prompt generation failed (exit=$PROMPT_EXIT)${NC}"
        echo "EXIT=$PROMPT_EXIT  TIMESTAMP=$(ts)" > "$staging/.phase_b_error"
    fi
done

echo ""
echo -e "${GREEN}[B] Phase B complete — all prompts generated${NC}"

# ── PHASE C: Ollama VLM description (all staging dirs) ───────────────────────
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  PHASE C: Ollama VLM Description (GPU)                  ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"

# Start Ollama server + pull model (one-time)
echo -e "${YELLOW}[$(ts)][C] Starting Ollama server (GPU)...${NC}"
ollama serve > /var/log/ollama.log 2>&1 &
OLLAMA_PID=$!

# Retry loop — Ollama can take 5-15s on cold start
OLLAMA_READY=false
for attempt in $(seq 1 10); do
    sleep 2
    if kill -0 $OLLAMA_PID 2>/dev/null && curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        OLLAMA_READY=true
        echo -e "${GREEN}[$(ts)][C] Ollama server ready (attempt $attempt)${NC}"
        break
    fi
    echo -e "${YELLOW}[$(ts)][C] Waiting for Ollama... (attempt $attempt/10)${NC}"
done

if [ "$OLLAMA_READY" = false ]; then
    echo -e "${RED}[$(ts)][C] ❌ Ollama server failed to start after 10 attempts${NC}"
    echo "=== Ollama log (last 30 lines) ===" > "$LOG_DIR/ollama_error.log"
    tail -30 /var/log/ollama.log >> "$LOG_DIR/ollama_error.log"
    exit 1
fi

# Pull model if not cached
echo -e "${YELLOW}[$(ts)][C] Pulling VLM model: qwen2.5vl:7b${NC}"
ollama pull qwen2.5vl:7b 2>&1 | tee "$LOG_DIR/ollama_pull.log"
PULL_EXIT=${PIPESTATUS[0]}
if [ $PULL_EXIT -ne 0 ]; then
    echo -e "${RED}[$(ts)][C] ❌ Model pull failed (exit=$PULL_EXIT)${NC}"
    exit 1
fi
echo -e "${GREEN}[$(ts)][C] Model ready${NC}"

# Verify GPU usage with actual inference test
echo -e "${YELLOW}[$(ts)][C] Verifying GPU inference...${NC}"
ollama run qwen2.5vl:7b "test" --verbose 2>&1 | head -5 | tee "$LOG_DIR/ollama_gpu_test.log"
sleep 1
echo -e "${CYAN}[$(ts)][C] GPU stats after inference test:${NC}"
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader | tee -a "$LOG_DIR/ollama_gpu_test.log"

# Process each book — skip dirs without prompt_config.json (Phase B failed)
for staging in "$STAGING_DIR"/*/; do
    BOOKNAME=$(basename "$staging")

    if [ ! -f "$staging/prompt_config.json" ]; then
        echo -e "${RED}[$(ts)][C] ⏭️  $BOOKNAME: skipped (prompt_config.json missing — Phase B failed)${NC}"
        continue
    fi

    echo ""
    echo -e "${YELLOW}[$(ts)][C] Describing figures for: $BOOKNAME${NC}"

    set +e
    python3 /workspace/vlm_describe.py "$staging" --max-workers 2 2>&1 | tee "$LOG_DIR/${BOOKNAME}_vlm.log"
    VLM_EXIT=$?
    set -e

    # Copy enhanced markdown to output
    if [ $VLM_EXIT -eq 0 ] && [ -f "$staging/${BOOKNAME}_enhanced.md" ]; then
        cp "$staging/${BOOKNAME}_enhanced.md" "$OUTPUT_DIR/"
        echo -e "${GREEN}[$(ts)][C] ✅ $BOOKNAME: _enhanced.md → output/${NC}"
    else
        echo -e "${RED}[$(ts)][C] ❌ $BOOKNAME: VLM description failed (exit=$VLM_EXIT)${NC}"
        echo "EXIT=$VLM_EXIT  TIMESTAMP=$(ts)" > "$staging/.phase_c_error"
    fi
done

# Stop Ollama
kill $OLLAMA_PID 2>/dev/null || true
echo ""
echo -e "${GREEN}[C] Phase C complete — all VLM descriptions done${NC}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ PIPELINE COMPLETE                                    ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Output files:"
find "$OUTPUT_DIR" -name "*_enhanced.md" -exec ls -lh {} \; 2>/dev/null || echo "  (none)"
echo ""
echo "Logs: $LOG_DIR/"
ls -la "$LOG_DIR/"
echo ""
echo "DONE" > "$OUTPUT_DIR/DONE"
date >> "$OUTPUT_DIR/DONE"
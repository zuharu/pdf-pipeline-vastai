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
    echo -e "${YELLOW}[A] Extracting: $BOOKNAME${NC}"

    # marker_single: --output_dir is a FLAG (not positional!) — Context7 verified
    marker_single "$pdf" \
        --output_dir "$STAGING" \
        --force_ocr \
        2>&1 | tee "$LOG_DIR/${BOOKNAME}_marker.log"

    # Verify output
    if [ -f "$STAGING/$BOOKNAME.md" ]; then
        JPEG_COUNT=$(ls -1 "$STAGING"/_page_*.jp*g 2>/dev/null | wc -l)
        echo -e "${GREEN}[A] ✅ $BOOKNAME: $(wc -c < "$STAGING/$BOOKNAME.md") bytes, $JPEG_COUNT figures${NC}"
    else
        echo -e "${RED}[A] ❌ $BOOKNAME: marker-pdf failed (no .md output)${NC}"
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
    echo ""
    echo -e "${YELLOW}[B] Generating prompts for: $BOOKNAME${NC}"

    python3 /workspace/vlm_prompt_gen.py "$staging" 2>&1 | tee "$LOG_DIR/${BOOKNAME}_prompt_gen.log"

    if [ -f "$staging/prompt_config.json" ]; then
        echo -e "${GREEN}[B] ✅ $BOOKNAME: prompt_config.json ready${NC}"
    else
        echo -e "${RED}[B] ❌ $BOOKNAME: prompt generation failed${NC}"
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
echo -e "${YELLOW}[C] Starting Ollama server...${NC}"
ollama serve > /var/log/ollama.log 2>&1 &
OLLAMA_PID=$!
sleep 3

# Verify ollama is running
if ! kill -0 $OLLAMA_PID 2>/dev/null; then
    echo -e "${RED}[C] ❌ Ollama server failed to start${NC}"
    cat /var/log/ollama.log | tail -20
    exit 1
fi
echo -e "${GREEN}[C] Ollama server running (PID $OLLAMA_PID)${NC}"

# Pull model if not cached
echo -e "${YELLOW}[C] Pulling VLM model: qwen2.5vl:7b${NC}"
ollama pull qwen2.5vl:7b 2>&1 | tee "$LOG_DIR/ollama_pull.log"
echo -e "${GREEN}[C] Model ready${NC}"

# Verify GPU usage
echo -e "${YELLOW}[C] Verifying GPU usage...${NC}"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# Process each book
for staging in "$STAGING_DIR"/*/; do
    BOOKNAME=$(basename "$staging")
    echo ""
    echo -e "${YELLOW}[C] Describing figures for: $BOOKNAME${NC}"

    python3 /workspace/vlm_describe.py "$staging" --max-workers 2 2>&1 | tee "$LOG_DIR/${BOOKNAME}_vlm.log"

    # Copy enhanced markdown to output
    if [ -f "$staging/${BOOKNAME}_enhanced.md" ]; then
        cp "$staging/${BOOKNAME}_enhanced.md" "$OUTPUT_DIR/"
        echo -e "${GREEN}[C] ✅ $BOOKNAME: _enhanced.md → output/${NC}"
    else
        echo -e "${RED}[C] ❌ $BOOKNAME: VLM description failed${NC}"
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
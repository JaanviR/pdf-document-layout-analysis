#!/bin/bash
# =============================================================================
# Parallel Processing Test Script
#
# Usage:
#   chmod +x test_parallel.sh
#   ./test_parallel.sh                  # full restart + test
#   ./test_parallel.sh --docs-only      # skip restart, just re-run with new docs
#
# Edit the DOCUMENTS and SETTINGS sections below, then run.
# Change RESTART to "false" if you only changed document names.
# =============================================================================

set -e

# --------------- DOCUMENTS ---------------
DOC1="../../books/test/doc2.pdf"
DOC2="../../books/test/doc3.pdf"

# --------------- SETTINGS ----------------
# Change these → service restarts automatically
PDF_WORKERS=2
INFERENCE_BATCH_SIZE=1
INFERENCE_NUM_WORKERS=0
CUDA_AMP=false

# --------------- OPTIONS -----------------
COMPOSE_FILE="docker-compose-gpu.yml"
RESULTS_DIR="../../test_results"
# ------------------------------------------

SERVICE="pdf-document-layout-analysis-gpu"
URL="http://localhost:5060"

# --docs-only flag skips restart
SKIP_RESTART=false
if [ "$1" = "--docs-only" ]; then
    SKIP_RESTART=true
fi

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "  Parallel Processing Test"
echo "========================================"
echo "  Doc 1:       $DOC1"
echo "  Doc 2:       $DOC2"
echo "  Workers:     $PDF_WORKERS"
echo "  Batch size:  $INFERENCE_BATCH_SIZE"
echo "  Num workers: $INFERENCE_NUM_WORKERS"
echo "  AMP:         $CUDA_AMP"
echo "========================================"
echo ""

# Verify docs exist
for doc in "$DOC1" "$DOC2"; do
    if [ ! -f "$doc" ]; then
        echo "ERROR: File not found: $doc"
        exit 1
    fi
done

# =========================================================================
# SERVICE STARTUP (skipped with --docs-only)
# =========================================================================
if [ "$SKIP_RESTART" = true ]; then
    echo "[skip] --docs-only: keeping current service running"
    echo ""
    # Quick check that service is actually up
    if ! curl -s "$URL/" > /dev/null 2>&1; then
        echo "ERROR: Service is not running. Run without --docs-only first."
        exit 1
    fi
else
    echo "[1/3] Restarting service..."
    docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true
    sleep 3

    PDF_WORKERS=$PDF_WORKERS \
    INFERENCE_BATCH_SIZE=$INFERENCE_BATCH_SIZE \
    INFERENCE_NUM_WORKERS=$INFERENCE_NUM_WORKERS \
    CUDA_AMP=$CUDA_AMP \
    docker compose -f "$COMPOSE_FILE" up -d 2>&1

    # Wait for service to respond
    echo "       Waiting for service..."
    for i in $(seq 1 300); do
        if curl -s "$URL/" > /dev/null 2>&1; then
            echo "       Service responding after ${i}s"
            break
        fi
        if [ "$i" -eq 300 ]; then
            echo "ERROR: Service did not start within 300s"
            docker compose -f "$COMPOSE_FILE" logs --tail 30 "$SERVICE"
            exit 1
        fi
        sleep 1
    done

    # Wait for warmup
    echo "       Waiting for worker warmup..."
    for i in $(seq 1 300); do
        if docker logs "$SERVICE" 2>&1 | grep -q "All workers warmed up"; then
            echo "       Warmup complete."
            break
        fi
        if [ "$i" -eq 300 ]; then
            echo "       WARNING: Warmup not detected, proceeding..."
        fi
        sleep 1
    done
    echo ""
fi

# =========================================================================
# GPU MEMORY — IDLE (after warmup, before processing)
# =========================================================================
echo "[gpu] Memory after warmup (idle):"
GPU_IDLE=$(nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader)
echo "       $GPU_IDLE (used, free, total MiB)"
echo ""
echo "       Per process:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while read line; do
    echo "         $line"
done
echo ""

# =========================================================================
# SEQUENTIAL TEST
# =========================================================================
echo "[2/3] Sequential test (one after the other)..."

# Start GPU memory monitor in background
> /tmp/gpu_seq_samples.txt
(
    while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' >> /tmp/gpu_seq_samples.txt
        sleep 0.5
    done
) &
GPU_MON_PID=$!

SEQ_START=$(date +%s%N)

curl -s -X POST "$URL/" -F "file=@${DOC1}" > "$RESULTS_DIR/seq_1.json"
curl -s -X POST "$URL/" -F "file=@${DOC2}" > "$RESULTS_DIR/seq_2.json"

SEQ_END=$(date +%s%N)
kill $GPU_MON_PID 2>/dev/null; wait $GPU_MON_PID 2>/dev/null || true

SEQ_MS=$(( (SEQ_END - SEQ_START) / 1000000 ))
SEQ_SEC=$(echo "scale=2; $SEQ_MS / 1000" | bc)
SEQ_PEAK=$(sort -n /tmp/gpu_seq_samples.txt 2>/dev/null | tail -1 || echo "N/A")

SEG1=$(python3 -c "import json; print(len(json.load(open('$RESULTS_DIR/seq_1.json'))))" 2>/dev/null || echo "error")
SEG2=$(python3 -c "import json; print(len(json.load(open('$RESULTS_DIR/seq_2.json'))))" 2>/dev/null || echo "error")

echo "       Time:     ${SEQ_SEC}s"
echo "       Peak GPU: ${SEQ_PEAK} MiB"
echo "       Doc 1:    $SEG1 segments"
echo "       Doc 2:    $SEG2 segments"
echo ""

# =========================================================================
# PARALLEL TEST
# =========================================================================
echo "[3/3] Parallel test (both at the same time)..."

# Start GPU memory monitor in background
> /tmp/gpu_par_samples.txt
(
    while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' >> /tmp/gpu_par_samples.txt
        sleep 0.5
    done
) &
GPU_MON_PID=$!

PAR_START=$(date +%s%N)

curl -s -X POST "$URL/" -F "file=@${DOC1}" > "$RESULTS_DIR/par_1.json" &
PID1=$!
curl -s -X POST "$URL/" -F "file=@${DOC2}" > "$RESULTS_DIR/par_2.json" &
PID2=$!

wait $PID1
wait $PID2

PAR_END=$(date +%s%N)
kill $GPU_MON_PID 2>/dev/null; wait $GPU_MON_PID 2>/dev/null || true

PAR_MS=$(( (PAR_END - PAR_START) / 1000000 ))
PAR_SEC=$(echo "scale=2; $PAR_MS / 1000" | bc)
PAR_PEAK=$(sort -n /tmp/gpu_par_samples.txt 2>/dev/null | tail -1 || echo "N/A")

PSEG1=$(python3 -c "import json; print(len(json.load(open('$RESULTS_DIR/par_1.json'))))" 2>/dev/null || echo "error")
PSEG2=$(python3 -c "import json; print(len(json.load(open('$RESULTS_DIR/par_2.json'))))" 2>/dev/null || echo "error")

echo "       Time:     ${PAR_SEC}s"
echo "       Peak GPU: ${PAR_PEAK} MiB"
echo "       Doc 1:    $PSEG1 segments"
echo "       Doc 2:    $PSEG2 segments"
echo ""

# =========================================================================
# RESULTS
# =========================================================================
echo "========================================"
echo "  RESULTS"
echo "========================================"
echo ""
echo "  Settings:    workers=$PDF_WORKERS  batch=$INFERENCE_BATCH_SIZE  amp=$CUDA_AMP"
echo ""
echo "  Sequential:  ${SEQ_SEC}s   (peak GPU: ${SEQ_PEAK} MiB)"
echo "  Parallel:    ${PAR_SEC}s   (peak GPU: ${PAR_PEAK} MiB)"
echo "  GPU idle:    $GPU_IDLE"
echo ""

if [ "$SEQ_MS" -gt 0 ] && [ "$PAR_MS" -gt 0 ]; then
    SPEEDUP=$(echo "scale=1; $SEQ_MS / $PAR_MS" | bc)
    echo "  Speedup:     ${SPEEDUP}x"
    echo ""
fi

# Validate results
python3 -c "
import json

files = [
    ('$RESULTS_DIR/seq_1.json', '$RESULTS_DIR/par_1.json', 'Doc 1'),
    ('$RESULTS_DIR/seq_2.json', '$RESULTS_DIR/par_2.json', 'Doc 2'),
]

all_pass = True
for seq_f, par_f, label in files:
    try:
        seq = json.load(open(seq_f))
        par = json.load(open(par_f))
        if len(seq) == len(par):
            print(f'  {label}: PASS ({len(seq)} segments match)')
        else:
            print(f'  {label}: MISMATCH (sequential={len(seq)}, parallel={len(par)})')
            all_pass = False
    except Exception as e:
        print(f'  {label}: ERROR ({e})')
        all_pass = False

print()
if all_pass:
    print('  All results match. Parallel processing is working correctly.')
else:
    print('  WARNING: Some results differ. Check logs:')
    print('    docker logs $SERVICE 2>&1 | tail -30')
"

echo ""
echo "  Results saved to: $RESULTS_DIR/"
echo "========================================"

# Cleanup
rm -f /tmp/gpu_seq_samples.txt /tmp/gpu_par_samples.txt

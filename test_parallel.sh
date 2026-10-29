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
# =============================================================================

set -e

# --------------- DOCUMENTS ---------------
# Add or remove lines — works with any number of docs
DOCS=(
    "../../books/test/doc2.pdf"
    "../../books/test/doc3.pdf"
)

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
NUM_DOCS=${#DOCS[@]}

# --docs-only flag skips restart
SKIP_RESTART=false
if [ "$1" = "--docs-only" ]; then
    SKIP_RESTART=true
fi

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "  Parallel Processing Test"
echo "========================================"
echo "  Documents:   $NUM_DOCS"
for i in $(seq 0 $((NUM_DOCS - 1))); do
    echo "    [$((i+1))] ${DOCS[$i]}"
done
echo "  Workers:     $PDF_WORKERS"
echo "  Batch size:  $INFERENCE_BATCH_SIZE"
echo "  Num workers: $INFERENCE_NUM_WORKERS"
echo "  AMP:         $CUDA_AMP"
echo "========================================"
echo ""

# Verify docs exist
for doc in "${DOCS[@]}"; do
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
# GPU MEMORY — IDLE
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
echo "[2/3] Sequential test ($NUM_DOCS docs, one after the other)..."

> /tmp/gpu_seq_samples.txt
(
    while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' >> /tmp/gpu_seq_samples.txt
        sleep 0.5
    done
) &
GPU_MON_PID=$!

SEQ_START=$(date +%s%N)

for i in $(seq 0 $((NUM_DOCS - 1))); do
    curl -s -X POST "$URL/" -F "file=@${DOCS[$i]}" > "$RESULTS_DIR/seq_$((i+1)).json"
done

SEQ_END=$(date +%s%N)
kill $GPU_MON_PID 2>/dev/null; wait $GPU_MON_PID 2>/dev/null || true

SEQ_MS=$(( (SEQ_END - SEQ_START) / 1000000 ))
SEQ_SEC=$(echo "scale=2; $SEQ_MS / 1000" | bc)
SEQ_PEAK=$(sort -n /tmp/gpu_seq_samples.txt 2>/dev/null | tail -1 || echo "N/A")

echo "       Time:     ${SEQ_SEC}s"
echo "       Peak GPU: ${SEQ_PEAK} MiB"
for i in $(seq 0 $((NUM_DOCS - 1))); do
    SEGS=$(python3 -c "import json; print(len(json.load(open('$RESULTS_DIR/seq_$((i+1)).json'))))" 2>/dev/null || echo "error")
    echo "       Doc $((i+1)):    $SEGS segments"
done
echo ""

# =========================================================================
# PARALLEL TEST
# =========================================================================
echo "[3/3] Parallel test ($NUM_DOCS docs, all at the same time)..."

> /tmp/gpu_par_samples.txt
(
    while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' >> /tmp/gpu_par_samples.txt
        sleep 0.5
    done
) &
GPU_MON_PID=$!

PAR_START=$(date +%s%N)

PIDS=()
for i in $(seq 0 $((NUM_DOCS - 1))); do
    curl -s -X POST "$URL/" -F "file=@${DOCS[$i]}" > "$RESULTS_DIR/par_$((i+1)).json" &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait $pid
done

PAR_END=$(date +%s%N)
kill $GPU_MON_PID 2>/dev/null; wait $GPU_MON_PID 2>/dev/null || true

PAR_MS=$(( (PAR_END - PAR_START) / 1000000 ))
PAR_SEC=$(echo "scale=2; $PAR_MS / 1000" | bc)
PAR_PEAK=$(sort -n /tmp/gpu_par_samples.txt 2>/dev/null | tail -1 || echo "N/A")

echo "       Time:     ${PAR_SEC}s"
echo "       Peak GPU: ${PAR_PEAK} MiB"
for i in $(seq 0 $((NUM_DOCS - 1))); do
    PSEGS=$(python3 -c "import json; print(len(json.load(open('$RESULTS_DIR/par_$((i+1)).json'))))" 2>/dev/null || echo "error")
    echo "       Doc $((i+1)):    $PSEGS segments"
done
echo ""

# =========================================================================
# RESULTS
# =========================================================================
echo "========================================"
echo "  RESULTS"
echo "========================================"
echo ""
echo "  Settings:    workers=$PDF_WORKERS  batch=$INFERENCE_BATCH_SIZE  amp=$CUDA_AMP"
echo "  Documents:   $NUM_DOCS"
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

# Validate: compare sequential vs parallel for each doc
python3 -c "
import json

num_docs = $NUM_DOCS
results_dir = '$RESULTS_DIR'

all_pass = True
for i in range(1, num_docs + 1):
    seq_f = f'{results_dir}/seq_{i}.json'
    par_f = f'{results_dir}/par_{i}.json'
    try:
        seq = json.load(open(seq_f))
        par = json.load(open(par_f))
        if len(seq) == len(par):
            print(f'  Doc {i}: PASS ({len(seq)} segments match)')
        else:
            print(f'  Doc {i}: MISMATCH (sequential={len(seq)}, parallel={len(par)})')
            all_pass = False
    except Exception as e:
        print(f'  Doc {i}: ERROR ({e})')
        all_pass = False

print()
if all_pass:
    print('  All results match. Parallel processing is working correctly.')
else:
    print('  WARNING: Some results differ. Check logs:')
    print('    docker logs pdf-document-layout-analysis-gpu 2>&1 | tail -30')
"

echo ""
echo "  Results saved to: $RESULTS_DIR/"
echo "========================================"

rm -f /tmp/gpu_seq_samples.txt /tmp/gpu_par_samples.txt

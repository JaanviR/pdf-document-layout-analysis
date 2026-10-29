#!/bin/bash
# =============================================================================
# GPU Memory Measurement Script
# Run this on your EC2 instance to measure memory usage for each configuration.
#
# Usage:
#   chmod +x measure_gpu_memory.sh
#   ./measure_gpu_memory.sh /path/to/test.pdf
#
# Uses a small test PDF by default (test_pdfs/regular.pdf).
# Pass a larger PDF to see peak memory during real workloads.
# =============================================================================

set -e

PDF_FILE="${1:-test_pdfs/regular.pdf}"
COMPOSE_FILE="docker-compose-gpu.yml"
RESULTS_FILE="gpu_memory_results.txt"
SERVICE_NAME="pdf-document-layout-analysis-gpu"

if [ ! -f "$PDF_FILE" ]; then
    echo "Error: PDF file not found: $PDF_FILE"
    echo "Usage: ./measure_gpu_memory.sh /path/to/test.pdf"
    exit 1
fi

echo "=============================================" | tee "$RESULTS_FILE"
echo "GPU Memory Measurement" | tee -a "$RESULTS_FILE"
echo "Date: $(date)" | tee -a "$RESULTS_FILE"
echo "PDF: $PDF_FILE" | tee -a "$RESULTS_FILE"
echo "=============================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Get GPU info
echo "--- GPU Info ---" | tee -a "$RESULTS_FILE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Function to get current GPU memory usage
get_gpu_memory() {
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
}

# Function to wait for service to be ready
wait_for_service() {
    echo "  Waiting for service to be ready..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:5060/ > /dev/null 2>&1; then
            echo "  Service ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: Service did not start within 120s"
    return 1
}

# Function to measure peak memory during PDF processing
measure_processing() {
    local label="$1"
    local peak_used=0

    echo "  Sending PDF for processing..."

    # Start monitoring GPU memory in background
    local monitor_pid
    (
        while true; do
            local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
            if [ -n "$used" ] && [ "$used" -gt 0 ] 2>/dev/null; then
                echo "$used" >> /tmp/gpu_mem_samples.txt
            fi
            sleep 0.5
        done
    ) &
    monitor_pid=$!

    > /tmp/gpu_mem_samples.txt

    # Send PDF and wait for response
    local start_time=$(date +%s)
    curl -s -X POST http://localhost:5060/ -F "file=@${PDF_FILE}" > /tmp/measure_result.json 2>&1
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Stop monitoring
    kill $monitor_pid 2>/dev/null
    wait $monitor_pid 2>/dev/null

    # Find peak memory from samples
    if [ -f /tmp/gpu_mem_samples.txt ] && [ -s /tmp/gpu_mem_samples.txt ]; then
        peak_used=$(sort -n /tmp/gpu_mem_samples.txt | tail -1)
    fi

    # Get segment count from result
    local segments=$(python3 -c "import json; d=json.load(open('/tmp/measure_result.json')); print(len(d))" 2>/dev/null || echo "error")

    echo "  Processing time: ${duration}s"
    echo "  Peak GPU memory: ${peak_used} MiB"
    echo "  Segments detected: ${segments}"

    rm -f /tmp/gpu_mem_samples.txt
}

# Function to run one test configuration
run_test() {
    local label="$1"
    local workers="$2"
    local batch="$3"
    local num_workers="$4"
    local amp="$5"

    echo "=============================================" | tee -a "$RESULTS_FILE"
    echo "TEST: $label" | tee -a "$RESULTS_FILE"
    echo "  PDF_WORKERS=$workers  BATCH_SIZE=$batch  NUM_WORKERS=$num_workers  AMP=$amp" | tee -a "$RESULTS_FILE"
    echo "=============================================" | tee -a "$RESULTS_FILE"

    # Stop any running service
    docker compose -f "$COMPOSE_FILE" down > /dev/null 2>&1 || true
    sleep 3

    # Record memory before starting
    local mem_before=$(get_gpu_memory)
    echo "  GPU before start: ${mem_before} MiB (used,free,total)" | tee -a "$RESULTS_FILE"

    # Start service
    echo "  Starting service..."
    PDF_WORKERS=$workers \
    INFERENCE_BATCH_SIZE=$batch \
    INFERENCE_NUM_WORKERS=$num_workers \
    CUDA_AMP=$amp \
    docker compose -f "$COMPOSE_FILE" up -d > /dev/null 2>&1

    if ! wait_for_service; then
        echo "  FAILED: Service did not start. Checking logs..." | tee -a "$RESULTS_FILE"
        docker compose -f "$COMPOSE_FILE" logs --tail 20 "$SERVICE_NAME" 2>/dev/null | tee -a "$RESULTS_FILE"
        docker compose -f "$COMPOSE_FILE" down > /dev/null 2>&1 || true
        echo "" | tee -a "$RESULTS_FILE"
        return 1
    fi

    # Record memory after model loaded (idle)
    sleep 5
    local mem_idle=$(get_gpu_memory)
    echo "  GPU after model load (idle): ${mem_idle} MiB (used,free,total)" | tee -a "$RESULTS_FILE"

    # Process PDF and measure peak memory
    measure_processing "$label"

    # Record final memory after processing
    sleep 2
    local mem_after=$(get_gpu_memory)
    echo "  GPU after processing: ${mem_after} MiB (used,free,total)" | tee -a "$RESULTS_FILE"

    # Capture peak into results
    local peak=$(sort -n /tmp/gpu_mem_samples.txt 2>/dev/null | tail -1 || echo "N/A")
    {
        echo "  Processing time: ${duration:-N/A}s"
        echo "  Segments: ${segments:-N/A}"
        echo ""
    } | tee -a "$RESULTS_FILE"

    # Stop service
    docker compose -f "$COMPOSE_FILE" down > /dev/null 2>&1 || true
    sleep 3
}

echo ""
echo "Starting measurements... This will take a while."
echo "Each test: start service, wait for model load, process PDF, measure memory."
echo ""

# ---- TEST 1: Baseline (original settings) ----
run_test "Baseline (original)" 1 1 0 false

# ---- TEST 2: Batch size 2, no AMP ----
run_test "Batch size 2, no AMP" 1 2 2 false

# ---- TEST 3: Batch size 4, no AMP ----
run_test "Batch size 4, no AMP" 1 4 2 false

# ---- TEST 4: Batch size 1, with AMP ----
run_test "AMP only (batch 1)" 1 1 0 true

# ---- TEST 5: Batch size 2, with AMP ----
run_test "Batch size 2 + AMP" 1 2 2 true

# ---- TEST 6: Batch size 4, with AMP ----
run_test "Batch size 4 + AMP" 1 4 2 true

# ---- TEST 7: 2 Workers (will likely OOM) ----
run_test "2 Workers (batch 1, no AMP)" 2 1 0 false

echo "" | tee -a "$RESULTS_FILE"
echo "=============================================" | tee -a "$RESULTS_FILE"
echo "ALL TESTS COMPLETE" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "=============================================" | tee -a "$RESULTS_FILE"

echo ""
echo "Done! Full results are in: $RESULTS_FILE"
echo ""
echo "To view: cat $RESULTS_FILE"

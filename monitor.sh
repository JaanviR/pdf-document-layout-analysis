#!/bin/bash
# =============================================================================
# GPU & CPU Monitor — Live Terminal Dashboard
#
# Usage:
#   chmod +x monitor.sh
#   ./monitor.sh              # live view, refreshes every 2s
#   ./monitor.sh --once       # single snapshot, no refresh
#   ./monitor.sh --log        # append snapshots to monitor_log.csv
# =============================================================================

SERVICE="pdf-document-layout-analysis-gpu"
INTERVAL=2
MODE="live"

if [ "$1" = "--once" ]; then MODE="once"; fi
if [ "$1" = "--log" ]; then MODE="log"; fi

LOG_FILE="monitor_log.csv"

if [ "$MODE" = "log" ] && [ ! -f "$LOG_FILE" ]; then
    echo "timestamp,gpu_used_mib,gpu_free_mib,gpu_util_pct,cpu_pct,worker1_mib,worker2_mib,phase,active_doc" > "$LOG_FILE"
fi

# Track previous CPU idle for delta calculation
PREV_IDLE=0
PREV_TOTAL=0

get_cpu_usage() {
    # Read /proc/stat for accurate CPU usage between intervals
    local CPU_LINE=$(head -1 /proc/stat)
    local USER=$(echo "$CPU_LINE" | awk '{print $2}')
    local NICE=$(echo "$CPU_LINE" | awk '{print $3}')
    local SYSTEM=$(echo "$CPU_LINE" | awk '{print $4}')
    local IDLE=$(echo "$CPU_LINE" | awk '{print $5}')
    local IOWAIT=$(echo "$CPU_LINE" | awk '{print $6}')
    local IRQ=$(echo "$CPU_LINE" | awk '{print $7}')
    local SOFTIRQ=$(echo "$CPU_LINE" | awk '{print $8}')

    local TOTAL=$((USER + NICE + SYSTEM + IDLE + IOWAIT + IRQ + SOFTIRQ))
    local DIFF_IDLE=$((IDLE - PREV_IDLE))
    local DIFF_TOTAL=$((TOTAL - PREV_TOTAL))

    PREV_IDLE=$IDLE
    PREV_TOTAL=$TOTAL

    if [ "$DIFF_TOTAL" -gt 0 ] 2>/dev/null; then
        echo $(( (1000 * (DIFF_TOTAL - DIFF_IDLE) / DIFF_TOTAL + 5) / 10 ))
    else
        echo "0"
    fi
}

make_bar() {
    local VALUE=$1
    local MAX=$2
    local WIDTH=$3
    local CHAR_FILL="█"
    local CHAR_EMPTY="░"

    local FILLED=0
    if [ "$MAX" -gt 0 ] 2>/dev/null; then
        FILLED=$(( VALUE * WIDTH / MAX ))
    fi
    [ "$FILLED" -gt "$WIDTH" ] && FILLED=$WIDTH
    local EMPTY=$(( WIDTH - FILLED ))

    local BAR=""
    for i in $(seq 1 $FILLED 2>/dev/null); do BAR="${BAR}${CHAR_FILL}"; done
    for i in $(seq 1 $EMPTY 2>/dev/null); do BAR="${BAR}${CHAR_EMPTY}"; done
    echo "$BAR"
}

detect_phase() {
    local GPU_UTIL=$1
    local CPU_PCT=$2
    local LAST_LOG_LINE=$3

    # Check recent docker logs for phase hints
    local RECENT=$(docker logs "$SERVICE" --tail 5 2>&1)

    # Check if "Creating PDF images" was the last action (no "Predicted" after it)
    local LAST_ACTION=$(docker logs "$SERVICE" --tail 20 2>&1 | grep -E "(Creating PDF images|Predicted|Processing file)" | tail -1)

    if echo "$LAST_ACTION" | grep -q "Predicted"; then
        echo "idle|Waiting for next request"
        return
    fi

    if echo "$LAST_ACTION" | grep -q "Creating PDF images"; then
        if [ "$GPU_UTIL" -lt 20 ] 2>/dev/null; then
            echo "cpu|Converting PDF pages to images (CPU)"
            return
        fi
    fi

    # Infer from utilization patterns
    if [ "$GPU_UTIL" -ge 80 ] 2>/dev/null; then
        echo "gpu|GPU inference — processing page batches"
        return
    elif [ "$GPU_UTIL" -ge 20 ] 2>/dev/null && [ "$GPU_UTIL" -lt 80 ] 2>/dev/null; then
        echo "transition|Loading data / building word grids"
        return
    elif [ "$CPU_PCT" -ge 30 ] 2>/dev/null; then
        echo "cpu|CPU work — images or word grids"
        return
    else
        echo "idle|Idle"
        return
    fi
}

show_dashboard() {
    local NOW=$(date '+%Y-%m-%d %H:%M:%S')
    local NUM_CPUS=$(nproc 2>/dev/null || echo "4")

    # GPU stats
    local GPU_INFO=$(nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
    local GPU_USED=$(echo "$GPU_INFO" | awk -F', ' '{print $1}' | tr -d ' ')
    local GPU_FREE=$(echo "$GPU_INFO" | awk -F', ' '{print $2}' | tr -d ' ')
    local GPU_TOTAL=$(echo "$GPU_INFO" | awk -F', ' '{print $3}' | tr -d ' ')
    local GPU_UTIL=$(echo "$GPU_INFO" | awk -F', ' '{print $4}' | tr -d ' ')
    local GPU_TEMP=$(echo "$GPU_INFO" | awk -F', ' '{print $5}' | tr -d ' ')
    local GPU_POWER=$(echo "$GPU_INFO" | awk -F', ' '{print $6}' | tr -d ' ')

    [ -z "$GPU_USED" ] && GPU_USED="0"
    [ -z "$GPU_FREE" ] && GPU_FREE="0"
    [ -z "$GPU_TOTAL" ] && GPU_TOTAL="15360"
    [ -z "$GPU_UTIL" ] && GPU_UTIL="0"
    [ -z "$GPU_TEMP" ] && GPU_TEMP="?"
    [ -z "$GPU_POWER" ] && GPU_POWER="?"

    # CPU stats
    local CPU_PCT=$(get_cpu_usage)

    # Per-process GPU memory
    local PROCS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
    local NUM_PROCS=$(echo "$PROCS" | grep -c '[0-9]' 2>/dev/null || echo "0")
    local W1_MEM="0"
    local W2_MEM="0"
    if [ "$NUM_PROCS" -ge 1 ]; then
        W1_MEM=$(echo "$PROCS" | sed -n '1p' | awk -F', ' '{print $2}' | tr -d ' ')
    fi
    if [ "$NUM_PROCS" -ge 2 ]; then
        W2_MEM=$(echo "$PROCS" | sed -n '2p' | awk -F', ' '{print $2}' | tr -d ' ')
    fi
    [ -z "$W1_MEM" ] && W1_MEM="0"
    [ -z "$W2_MEM" ] && W2_MEM="0"

    # Worker status
    local W1_STATUS="idle"
    local W2_STATUS="—"
    [ "$W1_MEM" -gt 3000 ] 2>/dev/null && W1_STATUS="ACTIVE"
    if [ "$NUM_PROCS" -ge 2 ]; then
        W2_STATUS="idle"
        [ "$W2_MEM" -gt 3000 ] 2>/dev/null && W2_STATUS="ACTIVE"
    fi

    # Currently processing doc
    local ACTIVE_DOC=$(docker logs "$SERVICE" --tail 30 2>&1 | grep -E "(Processing file|Predicted)" | \
        awk '
        /Processing file/ {
            fname = $0
            gsub(/.*Processing file: /, "", fname)
            started = $1 " " $2
            gsub(/,.*/, "", started)
            doc = fname
            doc_start = started
        }
        /Predicted/ {
            doc = ""
            doc_start = ""
        }
        END {
            if (doc != "") print doc "|" doc_start
        }' 2>/dev/null)

    local DOC_NAME=$(echo "$ACTIVE_DOC" | cut -d'|' -f1)
    local DOC_START=$(echo "$ACTIVE_DOC" | cut -d'|' -f2)
    local DOC_ELAPSED=""
    if [ -n "$DOC_START" ]; then
        local START_EPOCH=$(date -d "$DOC_START" +%s 2>/dev/null)
        local NOW_EPOCH=$(date +%s)
        if [ -n "$START_EPOCH" ]; then
            local ELAPSED_SEC=$((NOW_EPOCH - START_EPOCH))
            local ELAPSED_MIN=$((ELAPSED_SEC / 60))
            local ELAPSED_REM=$((ELAPSED_SEC % 60))
            DOC_ELAPSED="${ELAPSED_MIN}m ${ELAPSED_REM}s"
        fi
    fi

    # Phase detection
    local PHASE_INFO=$(detect_phase "$GPU_UTIL" "$CPU_PCT" "")
    local PHASE_TYPE=$(echo "$PHASE_INFO" | cut -d'|' -f1)
    local PHASE_DESC=$(echo "$PHASE_INFO" | cut -d'|' -f2)

    # Phase indicator color hint
    local PHASE_ICON="○"
    case "$PHASE_TYPE" in
        cpu) PHASE_ICON="◆ CPU" ;;
        gpu) PHASE_ICON="◆ GPU" ;;
        transition) PHASE_ICON="◇ MIX" ;;
        idle) PHASE_ICON="○ IDLE" ;;
    esac

    # Build bars
    local GPU_MEM_BAR=$(make_bar "$GPU_USED" "$GPU_TOTAL" 30)
    local GPU_UTIL_BAR=$(make_bar "$GPU_UTIL" 100 20)
    local CPU_BAR=$(make_bar "$CPU_PCT" 100 20)

    # Display
    echo "┌─────────────────────────────────────────────────────────────────┐"
    echo "│  PDF Layout Analysis — Live Monitor          $NOW  │"
    echo "├─────────────────────────────────────────────────────────────────┤"
    echo "│                                                                 │"
    echo "│  ┌─ GPU (Tesla T4) ──────────────────────────────────────────┐  │"
    printf "│  │  Memory:  %5s / %5s MiB                               │  │\n" "$GPU_USED" "$GPU_TOTAL"
    printf "│  │  [%s]  %s MiB free          │  │\n" "$GPU_MEM_BAR" "$GPU_FREE"
    printf "│  │  Util:    [%s] %3s%%                      │  │\n" "$GPU_UTIL_BAR" "$GPU_UTIL"
    printf "│  │  Temp: %s°C  |  Power: %sW                              │  │\n" "$GPU_TEMP" "$GPU_POWER"
    echo "│  └──────────────────────────────────────────────────────────┘  │"
    echo "│                                                                 │"
    echo "│  ┌─ CPU ($NUM_CPUS vCPUs) ──────────────────────────────────────────────┐  │"
    printf "│  │  Usage:   [%s] %3s%%                      │  │\n" "$CPU_BAR" "$CPU_PCT"
    echo "│  └──────────────────────────────────────────────────────────┘  │"
    echo "│                                                                 │"
    echo "├─────────────────────────────────────────────────────────────────┤"
    printf "│  Workers: %s on GPU\n" "$NUM_PROCS"
    if [ "$NUM_PROCS" -ge 1 ]; then
        printf "│    Worker 1:  %6s MiB  [%s]\n" "$W1_MEM" "$W1_STATUS"
    fi
    if [ "$NUM_PROCS" -ge 2 ]; then
        printf "│    Worker 2:  %6s MiB  [%s]\n" "$W2_MEM" "$W2_STATUS"
    fi
    echo "│                                                                 │"
    echo "├─────────────────────────────────────────────────────────────────┤"
    printf "│  Phase:  %s\n" "$PHASE_ICON"
    printf "│          %s\n" "$PHASE_DESC"
    echo "│                                                                 │"
    if [ -n "$DOC_NAME" ]; then
        printf "│  Document: %s\n" "$DOC_NAME"
        if [ -n "$DOC_ELAPSED" ]; then
            printf "│  Elapsed:  %s\n" "$DOC_ELAPSED"
        fi
    else
        echo "│  Document: (none — idle)                                        │"
    fi
    echo "│                                                                 │"
    echo "├─────────────────────────────────────────────────────────────────┤"
    echo "│  Recent Activity:                                               │"
    local RECENT_LOGS=$(docker logs "$SERVICE" --tail 30 2>&1 | grep -E "(Processing file|Predicted|ERROR|OOM|SIGKILL|Creating PDF)" | tail -6)
    if [ -z "$RECENT_LOGS" ]; then
        echo "│    (no recent activity)                                         │"
    else
        echo "$RECENT_LOGS" | while read line; do
            local TSTAMP=$(echo "$line" | grep -oP '\d{2}:\d{2}:\d{2}' | head -1)
            local SHORT=$(echo "$line" | sed 's/.*\[INFO\] //' | sed 's/.*\[ERROR\] /ERR: /' | cut -c1-50)
            printf "│    %s  %s\n" "$TSTAMP" "$SHORT"
        done
    fi
    echo "│                                                                 │"
    echo "└─────────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  GPU low + CPU high = image conversion or word grids (CPU phase)"
    echo "  GPU high + CPU low = model inference (GPU phase)"
    echo "  Both low = idle or post-processing"

    # CSV logging
    if [ "$MODE" = "log" ]; then
        echo "$NOW,$GPU_USED,$GPU_FREE,$GPU_UTIL,$CPU_PCT,$W1_MEM,$W2_MEM,$PHASE_TYPE,$DOC_NAME" >> "$LOG_FILE"
    fi
}

if [ "$MODE" = "once" ]; then
    # Prime CPU reading
    get_cpu_usage > /dev/null
    sleep 1
    show_dashboard
elif [ "$MODE" = "log" ]; then
    get_cpu_usage > /dev/null
    echo "Logging to $LOG_FILE every ${INTERVAL}s (Ctrl+C to stop)"
    while true; do
        show_dashboard
        sleep $INTERVAL
    done
else
    get_cpu_usage > /dev/null
    echo "Live monitor (Ctrl+C to stop, refreshes every ${INTERVAL}s)"
    echo ""
    while true; do
        clear
        show_dashboard
        sleep $INTERVAL
    done
fi

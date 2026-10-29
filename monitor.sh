#!/bin/bash
# =============================================================================
# GPU & Service Monitor — Live Terminal Dashboard
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
    echo "timestamp,gpu_used_mib,gpu_free_mib,gpu_total_mib,gpu_util_pct,worker1_mib,worker2_mib,active_docs" > "$LOG_FILE"
fi

show_dashboard() {
    local NOW=$(date '+%Y-%m-%d %H:%M:%S')

    # GPU stats
    local GPU_INFO=$(nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
    local GPU_USED=$(echo "$GPU_INFO" | awk -F', ' '{print $1}' | tr -d ' ')
    local GPU_FREE=$(echo "$GPU_INFO" | awk -F', ' '{print $2}' | tr -d ' ')
    local GPU_TOTAL=$(echo "$GPU_INFO" | awk -F', ' '{print $3}' | tr -d ' ')
    local GPU_UTIL=$(echo "$GPU_INFO" | awk -F', ' '{print $4}' | tr -d ' ')
    local GPU_TEMP=$(echo "$GPU_INFO" | awk -F', ' '{print $5}' | tr -d ' ')
    local GPU_POWER=$(echo "$GPU_INFO" | awk -F', ' '{print $6}' | tr -d ' ')

    # Per-process GPU memory
    local PROCS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
    local W1_MEM=$(echo "$PROCS" | head -1 | awk -F', ' '{print $2}' | tr -d ' ')
    local W2_MEM=$(echo "$PROCS" | tail -1 | awk -F', ' '{print $2}' | tr -d ' ')
    local W1_PID=$(echo "$PROCS" | head -1 | awk -F', ' '{print $1}' | tr -d ' ')
    local W2_PID=$(echo "$PROCS" | tail -1 | awk -F', ' '{print $2}' | tr -d ' ')
    local NUM_WORKERS=$(echo "$PROCS" | grep -c '[0-9]')

    [ -z "$W1_MEM" ] && W1_MEM="0"
    [ -z "$W2_MEM" ] && W2_MEM="0"

    # Worker status based on memory
    local W1_STATUS="idle"
    local W2_STATUS="idle"
    [ "$W1_MEM" -gt 3000 ] 2>/dev/null && W1_STATUS="ACTIVE"
    [ "$W2_MEM" -gt 3000 ] 2>/dev/null && W2_STATUS="ACTIVE"

    # GPU memory bar
    local BAR_WIDTH=40
    local FILLED=0
    if [ "$GPU_TOTAL" -gt 0 ] 2>/dev/null; then
        FILLED=$(( GPU_USED * BAR_WIDTH / GPU_TOTAL ))
    fi
    local EMPTY=$(( BAR_WIDTH - FILLED ))
    local BAR=$(printf '%0.s█' $(seq 1 $FILLED 2>/dev/null) 2>/dev/null)
    local BAR_E=$(printf '%0.s░' $(seq 1 $EMPTY 2>/dev/null) 2>/dev/null)

    # Recent docker log activity (last 10 relevant lines)
    local RECENT_LOGS=$(docker logs "$SERVICE" --tail 30 2>&1 | grep -E "(Processing file|Predicted|ERROR|OOM|SIGKILL)" | tail -8)

    # Currently processing (files that started but haven't finished)
    local ACTIVE_DOCS=$(docker logs "$SERVICE" --tail 50 2>&1 | grep -E "(Processing file|Predicted)" | tail -20 | \
        awk '
        /Processing file/ {
            fname = $NF
            gsub(/.*Processing file: /, "", fname)
            started[fname] = $1 " " $2
        }
        /Predicted/ {
            for (f in started) delete started[f]
        }
        END {
            for (f in started) print f " (since " started[f] ")"
        }')

    local ACTIVE_COUNT=$(echo "$ACTIVE_DOCS" | grep -c '[a-zA-Z]' 2>/dev/null || echo "0")

    # Display
    echo "┌──────────────────────────────────────────────────────┐"
    echo "│  PDF Layout Analysis — GPU Monitor                   │"
    echo "│  $NOW                              │"
    echo "├──────────────────────────────────────────────────────┤"
    echo "│                                                      │"
    printf "│  GPU Memory:  %s/%s MiB  (%s%% util)\n" "$GPU_USED" "$GPU_TOTAL" "$GPU_UTIL"
    printf "│  [%s%s]\n" "$BAR" "$BAR_E"
    printf "│  Free: %s MiB  |  Temp: %s°C  |  Power: %sW\n" "$GPU_FREE" "$GPU_TEMP" "$GPU_POWER"
    echo "│                                                      │"
    echo "├──────────────────────────────────────────────────────┤"
    printf "│  Workers: %s active\n" "$NUM_WORKERS"
    printf "│    Worker 1:  %6s MiB  [%s]\n" "$W1_MEM" "$W1_STATUS"
    printf "│    Worker 2:  %6s MiB  [%s]\n" "$W2_MEM" "$W2_STATUS"
    echo "│                                                      │"
    echo "├──────────────────────────────────────────────────────┤"
    echo "│  Currently Processing:                               │"
    if [ -z "$ACTIVE_DOCS" ] || [ "$ACTIVE_COUNT" -eq 0 ]; then
        echo "│    (none — workers idle)                             │"
    else
        echo "$ACTIVE_DOCS" | while read line; do
            printf "│    → %s\n" "$line"
        done
    fi
    echo "│                                                      │"
    echo "├──────────────────────────────────────────────────────┤"
    echo "│  Recent Activity:                                    │"
    if [ -z "$RECENT_LOGS" ]; then
        echo "│    (no recent activity)                              │"
    else
        echo "$RECENT_LOGS" | while read line; do
            local SHORT=$(echo "$line" | sed 's/.*\[INFO\] //' | sed 's/.*\[ERROR\] /ERR: /' | cut -c1-52)
            local TSTAMP=$(echo "$line" | grep -oP '\d{2}:\d{2}:\d{2}' | head -1)
            printf "│    %s  %s\n" "$TSTAMP" "$SHORT"
        done
    fi
    echo "│                                                      │"
    echo "└──────────────────────────────────────────────────────┘"

    # CSV logging
    if [ "$MODE" = "log" ]; then
        echo "$NOW,$GPU_USED,$GPU_FREE,$GPU_TOTAL,$GPU_UTIL,$W1_MEM,$W2_MEM,$ACTIVE_COUNT" >> "$LOG_FILE"
    fi
}

if [ "$MODE" = "once" ]; then
    show_dashboard
elif [ "$MODE" = "log" ]; then
    echo "Logging to $LOG_FILE every ${INTERVAL}s (Ctrl+C to stop)"
    while true; do
        show_dashboard
        sleep $INTERVAL
    done
else
    echo "Live monitor (Ctrl+C to stop, refreshes every ${INTERVAL}s)"
    echo ""
    while true; do
        clear
        show_dashboard
        sleep $INTERVAL
    done
fi

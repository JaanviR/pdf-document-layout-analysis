#!/bin/bash
# =============================================================================
# Parse Docker Logs — Per-Document Processing Stats
#
# Usage:
#   chmod +x parse_logs.sh
#   ./parse_logs.sh                    # parse recent logs (last 500 lines)
#   ./parse_logs.sh --all              # parse all logs
#   ./parse_logs.sh --json             # output as JSON
#   docker logs pdf-document-layout-analysis-gpu 2>&1 | ./parse_logs.sh -
# =============================================================================

SERVICE="pdf-document-layout-analysis-gpu"
LINES="500"
FORMAT="table"
INPUT="docker"

for arg in "$@"; do
    case "$arg" in
        --all) LINES="all" ;;
        --json) FORMAT="json" ;;
        -) INPUT="stdin" ;;
    esac
done

if [ "$INPUT" = "stdin" ]; then
    LOGS=$(cat)
elif [ "$LINES" = "all" ]; then
    LOGS=$(docker logs "$SERVICE" 2>&1)
else
    LOGS=$(docker logs "$SERVICE" --tail "$LINES" 2>&1)
fi

echo "$LOGS" | python3 -c "
import sys
import re
import json
from datetime import datetime

lines = sys.stdin.readlines()

docs = []
current = {}
errors = []
warmups = 0

for line in lines:
    line = line.strip()

    # Match: Processing file
    m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[INFO\] Processing file: (.+)', line)
    if m:
        if current.get('file') and not current.get('end_time'):
            current['status'] = 'interrupted'
            current['segments'] = 0
            docs.append(current)
        current = {
            'file': m.group(2),
            'start_time': m.group(1),
            'status': 'processing',
        }
        continue

    # Match: Predicted N segments
    m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[INFO\] Predicted (\d+) segments', line)
    if m and current.get('file'):
        current['end_time'] = m.group(1)
        current['segments'] = int(m.group(2))
        current['status'] = 'success'

        start = datetime.strptime(current['start_time'], '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(current['end_time'], '%Y-%m-%d %H:%M:%S')
        current['duration_sec'] = (end - start).total_seconds()

        if current['file'] == 'warmup.pdf':
            warmups += 1
        docs.append(current)
        current = {}
        continue

    # Match: Error
    if 'ERROR' in line or 'OutOfMemoryError' in line:
        if current.get('file'):
            current['status'] = 'error'
            current['error'] = 'OOM' if 'OutOfMemory' in line else 'error'
            docs.append(current)
            current = {}
        continue

    # Match: SIGKILL
    if 'SIGKILL' in line:
        if current.get('file'):
            current['status'] = 'killed'
            current['error'] = 'OOM-killed by OS'
            docs.append(current)
            current = {}
        continue

# Handle any still-processing doc
if current.get('file') and not current.get('end_time'):
    current['status'] = 'in-progress'
    current['segments'] = '...'
    now = datetime.now()
    start = datetime.strptime(current['start_time'], '%Y-%m-%d %H:%M:%S')
    current['duration_sec'] = f'{(now - start).total_seconds():.0f}+ (running)'
    docs.append(current)

# Filter out warmups for display
real_docs = [d for d in docs if d['file'] != 'warmup.pdf']

format = '$FORMAT'

if format == 'json':
    print(json.dumps({'warmups': warmups, 'documents': real_docs}, indent=2, default=str))
else:
    print()
    print('=' * 90)
    print(f'  Document Processing Report  ({warmups} warmup requests filtered)')
    print('=' * 90)
    print()
    print(f'{\"File\":<35} {\"Status\":<12} {\"Segments\":>10} {\"Duration\":>12} {\"Started\":>18}')
    print('-' * 90)

    total_time = 0
    total_segments = 0
    success_count = 0
    error_count = 0

    for d in real_docs:
        name = d['file']
        if len(name) > 33:
            name = name[:30] + '...'

        status = d.get('status', '?')
        segments = d.get('segments', '?')
        duration = d.get('duration_sec', '?')
        started = d.get('start_time', '?')

        if isinstance(duration, (int, float)):
            if duration >= 60:
                dur_str = f'{duration/60:.1f}m'
            else:
                dur_str = f'{duration:.1f}s'
            total_time += duration
        else:
            dur_str = str(duration)

        if status == 'success':
            status_str = 'OK'
            success_count += 1
            if isinstance(segments, int):
                total_segments += segments
        elif status == 'error':
            status_str = 'ERROR'
            error_count += 1
        elif status == 'killed':
            status_str = 'OOM-KILL'
            error_count += 1
        elif status == 'in-progress':
            status_str = 'RUNNING'
        elif status == 'interrupted':
            status_str = 'INTERRUPT'
            error_count += 1
        else:
            status_str = status

        time_short = started.split(' ')[1] if ' ' in str(started) else started

        print(f'{name:<35} {status_str:<12} {str(segments):>10} {dur_str:>12} {time_short:>18}')

    print('-' * 90)
    print()
    print(f'  Total documents: {len(real_docs)}  |  Success: {success_count}  |  Errors: {error_count}')
    if total_time >= 60:
        print(f'  Total processing time: {total_time/60:.1f} minutes  |  Total segments: {total_segments}')
    else:
        print(f'  Total processing time: {total_time:.1f}s  |  Total segments: {total_segments}')
    if success_count > 0:
        avg = total_time / success_count
        print(f'  Average per document: {avg:.1f}s ({avg/60:.1f}m)')
    print()
    print('=' * 90)
"

# Testing Guide: Step-by-Step

Push the entire repo to your g4dn.xlarge EC2 instance, then follow these steps in order. Each step tests one thing at a time so you can isolate any issues.

---

## Prerequisites

SSH into your EC2 instance:

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

Go to the project directory:

```bash
cd /path/to/pdf-document-layout-analysis
```

Make sure Docker and docker compose are installed:

```bash
docker --version
docker compose version
```

Make sure NVIDIA drivers and nvidia-container-toolkit are installed:

```bash
nvidia-smi
```

You should see the Tesla T4 with 16 GB memory.

---

## Step 1: Build the Docker image

```bash
docker compose -f docker-compose-gpu.yml build
```

This takes ~10-15 minutes. It downloads model weights, installs dependencies, and compiles CUDA extensions.

Check the build succeeded:

```bash
docker images | grep pdf-document-layout-analysis
```

---

## Step 2: Start with ORIGINAL settings (baseline)

First, let's test with all optimizations OFF to get a baseline. This proves the code works before we turn on optimizations.

```bash
PDF_WORKERS=1 \
INFERENCE_BATCH_SIZE=1 \
INFERENCE_NUM_WORKERS=0 \
CUDA_AMP=false \
docker compose -f docker-compose-gpu.yml up -d
```

Wait for the service to be ready (model loading takes 1-2 minutes):

```bash
# Watch logs until you see "Uvicorn running"
docker compose -f docker-compose-gpu.yml logs -f pdf-document-layout-analysis-gpu
```

Press `Ctrl+C` when you see the server is ready.

Verify the service is running and GPU is detected:

```bash
curl http://localhost:5060/
```

You should see something like: `3.12.x Using GPU: True`

---

## Step 3: Baseline test — single small PDF

Send the test PDF that comes with the repo:

```bash
time curl -X POST http://localhost:5060/ \
  -F "file=@test_pdfs/regular.pdf" \
  > baseline_result.json
```

Check the output looks correct:

```bash
# Should show a JSON array of segments with type, text, bounding boxes
cat baseline_result.json | python3 -m json.tool | head -30
```

```bash
# Count how many segments were detected
cat baseline_result.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{len(d)} segments detected')"
```

Note the time. This is your baseline.

---

## Step 4: Test the /analyze endpoint (returns segments + XML)

```bash
time curl -X POST http://localhost:5060/analyze \
  -F "file=@test_pdfs/regular.pdf" \
  > baseline_analyze.json
```

```bash
# Check it returns both segments and XML
cat baseline_analyze.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Segments: {len(d[\"segments\"])}')
print(f'XML starts with: {d[\"xml\"][:50]}')
"
```

---

## Step 5: Stop the service

```bash
docker compose -f docker-compose-gpu.yml down
```

---

## Step 6: Test with BATCHING ONLY

This isolates the batching improvement. No AMP, no extra workers.

```bash
PDF_WORKERS=1 \
INFERENCE_BATCH_SIZE=4 \
INFERENCE_NUM_WORKERS=2 \
CUDA_AMP=false \
docker compose -f docker-compose-gpu.yml up -d
```

Wait for the service to be ready:

```bash
docker compose -f docker-compose-gpu.yml logs -f pdf-document-layout-analysis-gpu
# Wait for "Uvicorn running", then Ctrl+C
```

```bash
curl http://localhost:5060/
```

Run the same test:

```bash
time curl -X POST http://localhost:5060/ \
  -F "file=@test_pdfs/regular.pdf" \
  > batched_result.json
```

Compare results with baseline:

```bash
# Should be the same number of segments
cat batched_result.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{len(d)} segments detected')"
```

```bash
# Diff the two results (should be identical or very close)
python3 -c "
import json
a = json.load(open('baseline_result.json'))
b = json.load(open('batched_result.json'))
print(f'Baseline segments: {len(a)}')
print(f'Batched segments:  {len(b)}')
if len(a) == len(b):
    diffs = 0
    for i in range(len(a)):
        if a[i]['type'] != b[i]['type'] or a[i]['text'] != b[i]['text']:
            diffs += 1
            print(f'  Segment {i}: type {a[i][\"type\"]} vs {b[i][\"type\"]}, text match: {a[i][\"text\"] == b[i][\"text\"]}')
    if diffs == 0:
        print('PASS: All segments identical')
    else:
        print(f'WARNING: {diffs} segments differ')
else:
    print('WARNING: Different number of segments')
"
```

Monitor GPU during processing (run in another terminal):

```bash
watch -n 2 nvidia-smi
```

You should see higher GPU utilization than the baseline.

Stop the service:

```bash
docker compose -f docker-compose-gpu.yml down
```

---

## Step 7: Test with BATCHING + AMP

```bash
PDF_WORKERS=1 \
INFERENCE_BATCH_SIZE=4 \
INFERENCE_NUM_WORKERS=2 \
CUDA_AMP=true \
docker compose -f docker-compose-gpu.yml up -d
```

Wait for ready:

```bash
docker compose -f docker-compose-gpu.yml logs -f pdf-document-layout-analysis-gpu
# Wait for "Uvicorn running", then Ctrl+C
```

```bash
time curl -X POST http://localhost:5060/ \
  -F "file=@test_pdfs/regular.pdf" \
  > amp_result.json
```

Run the AMP quality comparison script:

```bash
python3 compare_amp_results.py baseline_result.json amp_result.json
```

This should print `VERDICT: PASS`. If it says FAIL, disable AMP (`CUDA_AMP=false`).

Stop:

```bash
docker compose -f docker-compose-gpu.yml down
```

---

## Step 8: Test with ALL optimizations (batching + AMP + 2 workers)

This is the full configuration:

```bash
PDF_WORKERS=2 \
INFERENCE_BATCH_SIZE=4 \
INFERENCE_NUM_WORKERS=2 \
CUDA_AMP=true \
docker compose -f docker-compose-gpu.yml up -d
```

Wait for ready (will take longer — 2 workers each load the model):

```bash
docker compose -f docker-compose-gpu.yml logs -f pdf-document-layout-analysis-gpu
# Wait until you see TWO "[INFO] Uvicorn running" messages, then Ctrl+C
```

Check GPU memory with both workers loaded:

```bash
nvidia-smi
```

Note the memory usage. Both workers should fit within 16 GB.

### Test single document (should work same as before):

```bash
time curl -X POST http://localhost:5060/ \
  -F "file=@test_pdfs/regular.pdf" \
  > full_result.json
```

```bash
python3 compare_amp_results.py baseline_result.json full_result.json
```

### Test parallel documents (the main goal):

Send 2 PDFs at the same time using background jobs:

```bash
time curl -X POST http://localhost:5060/ \
  -F "file=@test_pdfs/regular.pdf" \
  > parallel_result_1.json &
PID1=$!

time curl -X POST http://localhost:5060/ \
  -F "file=@test_pdfs/regular.pdf" \
  > parallel_result_2.json &
PID2=$!

echo "Both requests sent. Waiting for completion..."
wait $PID1
echo "Request 1 done"
wait $PID2
echo "Request 2 done"
```

While they're running, watch GPU in another terminal:

```bash
watch -n 2 nvidia-smi
```

You should see GPU utilization from both workers.

Verify neither result is corrupted:

```bash
python3 -c "
import json
r1 = json.load(open('parallel_result_1.json'))
r2 = json.load(open('parallel_result_2.json'))
b = json.load(open('baseline_result.json'))
print(f'Baseline:   {len(b)} segments')
print(f'Parallel 1: {len(r1)} segments')
print(f'Parallel 2: {len(r2)} segments')
if len(r1) == len(b) and len(r2) == len(b):
    print('PASS: Both parallel results have correct segment count')
else:
    print('WARNING: Segment counts differ — possible corruption')
"
```

---

## Step 9: Test with a LARGE PDF (the real test)

If you have the 800-page book or a large PDF, this is where you see the real impact:

```bash
# Single document with all optimizations
time curl -X POST http://localhost:5060/ \
  -F "file=@/path/to/your/large-document.pdf" \
  > large_result.json
```

Compare this time to the original ~4 hours. Should be ~1-1.5 hours.

For the parallel test with large docs:

```bash
time curl -X POST http://localhost:5060/ \
  -F "file=@/path/to/large-doc-1.pdf" \
  > large_parallel_1.json &

time curl -X POST http://localhost:5060/ \
  -F "file=@/path/to/large-doc-2.pdf" \
  > large_parallel_2.json &

wait
echo "Both done"
```

---

## Step 10: Stress test — check for OOM

Monitor memory while processing large docs:

```bash
# GPU memory (run in separate terminal)
watch -n 5 nvidia-smi

# System RAM (run in another separate terminal)
watch -n 5 free -h
```

If GPU runs out of memory, you'll see errors in the logs:

```bash
docker compose -f docker-compose-gpu.yml logs --tail 50 pdf-document-layout-analysis-gpu
```

Look for `CUDA out of memory` or `RuntimeError`. If you see it:

```bash
# Stop
docker compose -f docker-compose-gpu.yml down

# Restart with lower batch size
PDF_WORKERS=2 \
INFERENCE_BATCH_SIZE=2 \
INFERENCE_NUM_WORKERS=2 \
CUDA_AMP=true \
docker compose -f docker-compose-gpu.yml up -d
```

If system RAM runs out (container gets killed):

```bash
# Check if container was OOM-killed
docker inspect pdf-document-layout-analysis-gpu | grep -i oom

# Restart with fewer data workers
PDF_WORKERS=2 \
INFERENCE_BATCH_SIZE=4 \
INFERENCE_NUM_WORKERS=0 \
CUDA_AMP=true \
docker compose -f docker-compose-gpu.yml up -d
```

---

## Step 11 (Optional): Push batch size higher

If Step 10 shows plenty of GPU memory headroom:

```bash
docker compose -f docker-compose-gpu.yml down

PDF_WORKERS=2 \
INFERENCE_BATCH_SIZE=8 \
INFERENCE_NUM_WORKERS=2 \
CUDA_AMP=true \
docker compose -f docker-compose-gpu.yml up -d
```

Test with a large PDF and compare time to batch_size=4. If it's faster and no OOM, keep it.

---

## Quick Reference: Final Recommended Settings

```bash
PDF_WORKERS=2 \
INFERENCE_BATCH_SIZE=4 \
INFERENCE_NUM_WORKERS=2 \
CUDA_AMP=true \
docker compose -f docker-compose-gpu.yml up -d
```

Or just use the defaults (already set in docker-compose-gpu.yml):

```bash
docker compose -f docker-compose-gpu.yml up -d
```

---

## Troubleshooting

| Problem | What to check | Fix |
|---|---|---|
| `CUDA out of memory` | `nvidia-smi` shows full GPU | Lower `INFERENCE_BATCH_SIZE` to 2, or `PDF_WORKERS` to 1 |
| Container keeps restarting | `docker logs pdf-document-layout-analysis-gpu` | Check for errors in logs |
| Container OOM-killed | `docker inspect ... \| grep oom` | Lower `INFERENCE_NUM_WORKERS` to 0, or upgrade instance |
| Service very slow to start | Normal — model loading | Wait 2-3 minutes, watch logs |
| `Using GPU: False` in curl output | NVIDIA runtime not configured | Check `nvidia-smi` works, check nvidia-container-toolkit |
| Results differ between baseline and batched | Shouldn't happen | Report — may need to disable batching |
| AMP comparison shows FAIL | Model sensitive to FP16 | Set `CUDA_AMP=false` |

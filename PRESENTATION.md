# Huridocs PDF Processing: Performance Optimization

What we changed, why, and the expected impact on document processing times.

---

## The Problem

**Current state:** Documents process one at a time. An 800-page book takes ~4 hours on GPU. If a user uploads 5 documents, document #5 waits up to 16+ hours before it even starts.

The Huridocs PDF layout analysis service uses a machine learning model (VGT — Vision Grid Transformer) to detect the structure of each page in a PDF: tables, headings, paragraphs, figures, etc. The model processes each page individually by default.

Three issues compound to create the bottleneck:

1. **File collisions prevent parallelism.** All documents share global file paths for images, word grids, annotations, and model output. Processing two documents simultaneously would overwrite each other's files and produce corrupt results.
2. **A threading lock forces sequential processing.** To prevent file collisions, the code uses a `threading.Lock()` that allows only one document through the ML pipeline at a time.
3. **The GPU is underutilized.** Pages process one at a time (batch size = 1), GPU auto-tuning is disabled during inference, and full 32-bit precision is used for all calculations.

```
How it worked before:

Doc A (800 pages) ──── 4 hours ────▶ done
Doc B (800 pages)                     ── waits ── ── 4 hours ──▶ done
Doc C (10 pages)                                                ── waits ──▶

Total: ~8+ hours, documents stuck in queue
```

---

## Phase 1: Document Isolation

**What:** Each document gets its own private temporary workspace instead of sharing global paths.

```
Per-document workspace:

/tmp/pdla_a1b2c3d4/          ← unique per document
    images/                   ← page JPEGs for this document only
    word_grids/               ← word grid pickles for this document only
    jsons/                    ← COCO annotations for this document only
    output/                   ← model results for this document only

Workspace is auto-deleted when the document finishes.
```

**Why this matters:** Without isolation, parallel processing is impossible — two documents would overwrite each other's data. This phase makes parallelism *safe*. It doesn't make anything faster by itself.

### Files changed

| File | What changed |
|------|-------------|
| `PdfImages.py` | Images save to workspace, not global dir |
| `create_word_grid.py` | Word grids save to workspace |
| `get_json_annotations.py` | COCO JSON saves to workspace |
| `get_most_probable_pdf_segments.py` | Reads results from workspace |
| `vgt_model_adapter.py` | Orchestrates workspace creation + unique dataset names |
| `pdf_analysis_service_adapter.py` | Creates temp workspace per request |

All new parameters default to `None`, falling back to the old global paths. Existing tests, the Gradio GUI, and any external code that calls these functions without the new parameters works exactly as before.

---

## Phase 2: Inference Speed Optimizations

**What:** Three GPU optimizations that make each individual document process faster.

### The three optimizations

| Optimization | Before | After | Impact |
|---|---|---|---|
| **Batch Size** | 1 page at a time | 4 pages at a time | ~200 GPU passes instead of 800 for an 800-page PDF |
| **cuDNN Benchmark** | Off during inference | On | Auto-tunes convolution algorithms. ~10-15% faster per batch |
| **Mixed Precision (AMP)** | FP32 (32-bit) | FP16 (16-bit) | Heavy math at half precision. ~1.5-2x faster, half GPU memory |

### Files changed

| File | What changed |
|------|-------------|
| `VGTTrainer.py` | Batch size + num_workers from env vars, cudnn.benchmark enabled |
| `vgt_model_adapter.py` | AMP autocast wrapper around inference |
| `docker-compose-gpu.yml` | New environment variables |

### New environment variables

| Variable | Value | What it does |
|---|---|---|
| `INFERENCE_BATCH_SIZE` | 4 | Pages processed per GPU pass. Increase to 8 if GPU memory allows. |
| `INFERENCE_NUM_WORKERS` | 2 | CPU threads preparing data while GPU computes. |
| `CUDA_AMP` | true | Enables FP16 mixed precision. Only activates if GPU is present. |

> **Quality check:** AMP (mixed precision) uses FP16 for matrix math and FP32 for sensitive operations (normalization, softmax). A comparison script (`compare_amp_results.py`) is included — run the same PDF with AMP off and on, and the script verifies identical segment types, text content, and bounding boxes within rounding tolerance.

### Batching alone — the single biggest win

Even without AMP or cudnn.benchmark, just changing batch size from 1 to 4:

- ~90% of the 4-hour processing time is GPU inference = ~3.5 hours
- ~10% is CPU work (rendering images, word grids) = ~25 minutes
- Batch size 4 means 200 GPU passes instead of 800
- Realistic speedup: **2.5-3x** (not a clean 4x — each batch takes slightly longer due to memory bandwidth)
- Inference drops from ~3.5 hours to ~1.2-1.4 hours
- **Total with batching alone: ~1.5-1.7 hours** (down from ~4 hours)

---

## Phase 3: Parallel Document Processing

**What:** Increased Gunicorn from 1 worker to 2 workers. Each worker is a separate OS process with its own model copy.

```
How it works now:

Your RAG Agent
      |
  Gunicorn (port 5060)     ← same port, same container, zero changes to your agent
      |
      |  routes to a free worker
  ┌──────────┬──────────┐
  │ Worker 1 │ Worker 2 │     ← separate processes, each with own model
  │ (Doc A)  │ (Doc B)  │     ← processing at the same time
  └──────────┴──────────┘
        shared GPU (CUDA time-slices between them)
```

### Files changed

| File | What changed |
|------|-------------|
| `docker-compose-gpu.yml` | Added `-w ${PDF_WORKERS:-2}` to Gunicorn entrypoint |

**Why this is safe:** Phase 1 gave each document its own workspace (no file collisions). Each Gunicorn worker is a separate Python process — separate model, separate memory, separate lock. Two documents processing simultaneously never interfere.

**Why not nginx + multiple containers?** With a single GPU on g4dn.xlarge, Gunicorn workers achieve the same parallelism with less complexity. Nginx would only help when scaling to multiple machines or multiple GPUs.

| Variable | Default | What it does |
|---|---|---|
| `PDF_WORKERS` | 2 | Number of Gunicorn workers. Each loads its own model (~3 GB GPU memory per worker). |

---

## Upper Limits: How Far Can You Push It?

### Batch size — limited by GPU memory

| Batch Size | Approx GPU Memory | T4 (16 GB) |
|---|---|---|
| 1 | ~2 GB | Works |
| 4 | ~3-4 GB | Works (our default) |
| 8 | ~5-7 GB | Probably works |
| 16 | ~10-14 GB | Risky, might OOM |
| 32 | ~20+ GB | Won't fit |

Speedup has diminishing returns. Going from 1 → 4 saves the most. Beyond 8-16, the GPU cores are fully saturated — you're just filling memory for minimal gain.

### Gunicorn workers — limited by GPU memory + system RAM

Each worker loads its own model copy (~1.5 GB GPU + ~2 GB system RAM).

| Workers | GPU Memory (models) | System RAM | Feasible on g4dn.xlarge? |
|---|---|---|---|
| 1 | ~1.5 GB | ~2 GB | Yes |
| 2 | ~3 GB | ~4 GB | Yes (our default) |
| 3 | ~4.5 GB | ~6 GB | Probably, tight on system RAM |
| 4 | ~6 GB | ~8 GB | Risky — only 16 GB system RAM total |

More workers doesn't mean proportionally faster — they all share the same GPU compute cores. With 3 workers, each document gets 1/3 of GPU time instead of 1/2. Individual documents get **slower**, you just process **more concurrently**.

### Sweet spot for g4dn.xlarge (current infra)

```
PDF_WORKERS=2
INFERENCE_BATCH_SIZE=4   (try 8 if stable)
```

---

## Scaling with a Bigger GPU

| GPU | Memory | Batch Size | Workers | Time per 800-page doc | 2 docs parallel |
|---|---|---|---|---|---|
| **T4** (g4dn.xlarge, current) | 16 GB | 4 | 2 | ~1-1.5 hrs | ~1.5-2 hrs |
| **A10G** (g5.xlarge) | 24 GB | 8 | 3 | ~1-1.5 hrs | ~1.5-2 hrs |
| **A100** (p4d.xlarge) | 40 GB | 16 | 4-5 | ~30-45 min | ~40-60 min |
| **A100 80GB** (p4d) | 80 GB | 16 | 8-10 | ~30-45 min | ~35-50 min |

**Bigger GPU = faster cores too**, not just more memory. The A100 has ~5x the raw compute of a T4. So each batch also runs faster.

**Batch size caps out around 16.** Beyond that, GPU cores are fully saturated — more pages per batch just fills memory for no gain.

**Workers cap out around 4-5 on a single GPU.** After that, time-slicing overhead eats into gains. Better to go multi-GPU than more workers on one GPU.

### Best bang-for-buck upgrade

**g5.xlarge (A10G, 24 GB)** — ~$1.01/hr vs $0.53/hr (roughly double the cost):
- ~2x faster compute + 50% more memory
- Can run batch_size=8 with 3 workers comfortably
- 800-page book drops to ~1 hour, two books in parallel ~1.5 hours

Going to A100 is great but expensive (~$3-5/hr). Worth it only if processing documents constantly, not in bursts.

---

## Expected Impact

### Before (original)

- 1 document at a time
- ~4 hours per 800-page PDF
- 2 books = ~8 hours total
- Small docs wait hours behind big ones

### After (all optimizations, on current g4dn.xlarge)

- 2 documents in parallel
- ~1-1.5 hours per 800-page PDF
- 2 books = ~1.5-2 hours total
- Small docs process immediately on free worker

Infrastructure: g4dn.xlarge — Tesla T4 (16 GB GPU), 4 vCPUs, 16 GB system RAM.

---

## Risk & Rollback

**GPU memory:** 2 workers + batch_size=4 + AMP needs ~6 GB of the T4's 16 GB. Comfortable headroom. If memory issues arise, lower `INFERENCE_BATCH_SIZE` to 2 first, then `PDF_WORKERS` to 1 as last resort.

**System RAM:** g4dn.xlarge has only 16 GB system RAM with 4 vCPUs. Two workers each loading the model + BERT tokenizer into CPU memory could be tight. If the container gets OOM-killed, options are: lower batch size, lower num_workers to 0, or upgrade to g4dn.2xlarge (32 GB RAM).

**Rollback is simple.** Every optimization is controlled by an environment variable. To revert to original behavior:

| Variable | Revert value |
|---|---|
| `PDF_WORKERS` | 1 |
| `INFERENCE_BATCH_SIZE` | 1 |
| `INFERENCE_NUM_WORKERS` | 0 |
| `CUDA_AMP` | false |

No code changes needed to roll back — just change the environment variables and restart the container.

---

## Testing Plan

1. **Build & start:** `docker compose -f docker-compose-gpu.yml build && docker compose -f docker-compose-gpu.yml up -d`
2. **Smoke test:** Send a small PDF, verify output matches expected segments
3. **AMP quality check:** Run same PDF with `CUDA_AMP=false` then `CUDA_AMP=true`, compare with `compare_amp_results.py`
4. **Parallel test:** Send 2 PDFs simultaneously, verify both complete and neither has corrupted output
5. **Monitor:** Watch `nvidia-smi` during processing — GPU utilization should be higher, memory should stay within bounds

---

## Summary

| Phase | Change | Effect | Risk |
|---|---|---|---|
| **1. Isolation** | Per-document temp workspaces | Makes parallel processing safe | Low — backwards compatible, all defaults preserved |
| **2. Speed** | Batching, cuDNN auto-tune, FP16 | ~3-4x faster per document | Low — all controlled by env vars, rollback instant |
| **3. Parallel** | 2 Gunicorn workers | 2 documents process simultaneously | Medium — depends on GPU/system memory fitting 2 workers |

**Before all changes:** 2 books x 4 hours each = **8 hours total** (sequential)
**After all changes:** 2 books x ~1.5 hours each = **~1.5-2 hours total** (parallel, with speed optimizations)

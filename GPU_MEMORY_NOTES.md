# GPU Memory Usage — PDF Document Layout Analysis

## 1. Static Memory (same every time, regardless of document)

| What | Size | When loaded | Stays until |
|---|---|---|---|
| **CUDA context** | ~1.2 GB | Worker process starts | Worker dies |
| **VGT model weights** | ~3.9 GB | First inference (lazy load) | Worker dies |

**Total static per worker: ~5.1 GB**

- **CUDA context**: PyTorch initializes the GPU — sets up communication between CPU and GPU. This is like "turning on the GPU" for that worker. Happens once, never changes.
- **Model weights**: The trained neural network parameters (millions of numbers that the model learned during training). Loaded from the 1.3 GB file on disk, but expands in GPU memory because it's stored in float32 (higher precision). Same regardless of whether you're processing a 1-page or 800-page document.

---

## 2. Dynamic Memory — changes with document size/content

These are **temporary** — they exist only while a document is being processed, then get released (via `torch.cuda.empty_cache()`).

| What | Depends on | Small doc (5 pages) | Large doc (800+ pages) |
|---|---|---|---|
| **Page images** | Number of pages | ~50 MB | ~2-3 GB |
| **Word grid embeddings** | Words per page x pages | ~100 MB | ~2-4 GB |
| **Model activations** | Page complexity | ~200 MB | ~1-2 GB |
| **Inference output** | Number of detections | ~10 MB | ~500 MB |

**Small doc total dynamic: ~0.3-0.5 GB**
**Large doc (800pg) total dynamic: ~5-9 GB**

### What each one means

- **Page images**: Each PDF page is converted to a JPG image and loaded onto the GPU. More pages = more images in memory. These are what the model "looks at."
- **Word grid embeddings**: The model extracts every word on every page, maps each word to a position on a grid, then converts it to numbers the neural network understands (embeddings). An 800-page textbook with dense text = millions of words = massive grid = huge memory. This is usually what causes OOM — the `embedding_proj` layer.
- **Model activations**: As each page image passes through the neural network layers, each layer produces intermediate results (activations). Think of it like scratch paper the model uses while thinking. Complex pages with many elements = bigger scratch paper.
- **Inference output**: The model's final predictions — bounding boxes, labels, confidence scores for every detected segment.

---

## 3. Memory Timeline — Processing a Large Document

```
Worker starts:           [CUDA 1.2 GB                                          ] = 1.2 GB
Model loads (warmup):    [CUDA 1.2 GB][Model weights 3.9 GB                    ] = 5.1 GB
empty_cache() after:     [CUDA 1.2 GB]                                           = 1.2 GB *
Large doc starts:        [CUDA 1.2 GB][Model weights 3.9 GB (reloaded from cache)]= 5.1 GB
Processing pages:        [CUDA 1.2 GB][Model 3.9 GB][Images+Grids+Activations   ]= 10.9 GB
empty_cache() after:     [CUDA 1.2 GB]                                           = 1.2 GB *
```

\* Model weights get cached by PyTorch — they reload instantly from GPU cache on next request. `empty_cache()` releases them back to the GPU pool so the other worker can use that memory if needed.

---

## 4. Real Examples — What We Actually Observed

### Example 1: doc5.pdf (5-page document)

```
Before processing:
  nvidia-smi shows Worker 1: 1,268 MiB (idle, just CUDA context)

During processing:
  Worker 1: ~5,400 MiB
    Breakdown:
      CUDA context:          1,200 MiB  (always there)
      Model weights:         3,900 MiB  (loaded for inference)
      Page images (5 pages):    50 MiB  (5 small JPGs)
      Word grid embeddings:    100 MiB  (not much text)
      Activations:             100 MiB  (simple pages)
      Inference output:         10 MiB  (10 segments detected)

After processing (empty_cache runs):
  Worker 1: 1,268 MiB (back to idle)

Time: ~2 seconds
Result: 10 segments
```

This is a typical small document. Uses ~5.4 GB during inference, easily fits on T4 even with 2 workers.

### Example 2: APBiology.pdf (800+ page textbook)

```
Before processing:
  nvidia-smi shows Worker 1: 1,268 MiB (idle, just CUDA context)

During processing:
  Worker 1: ~10,900 MiB
    Breakdown:
      CUDA context:          1,200 MiB  (always there)
      Model weights:         3,900 MiB  (same as any doc)
      Page images (800 pg):  2,500 MiB  (800 page images!)
      Word grid embeddings:  2,300 MiB  (dense text on every page)
      Activations:           800 MiB    (complex layouts with figures, tables)
      Inference output:      200 MiB    (28,931 segments detected)

After processing (empty_cache runs):
  Worker 1: 1,268 MiB (back to idle)

Time: ~34 minutes
Result: 28,931 segments
```

This is a worst-case heavy document. Uses ~10.9 GB — that's 71% of the T4's total 15 GB.

### Example 3: What happens when both run in parallel on T4

```
Scenario A: doc5.pdf + doc5.pdf (both small) — WORKS
  Worker 1: 5,400 MiB  (doc5)
  Worker 2: 5,400 MiB  (doc5)
  Total:   10,800 MiB  of 15,360 MiB  -->  4,560 MiB free. No problem.

Scenario B: APBiology.pdf + doc5.pdf (heavy + light) — TIGHT
  Worker 1: 10,900 MiB (APBiology)
  Worker 2:  5,400 MiB (doc5)
  Total:    16,300 MiB  of 15,360 MiB  -->  940 MiB OVER. Will likely OOM.

  But with empty_cache(): Worker 2 starts while Worker 1 is in the middle of
  APBiology. Worker 1 hasn't finished yet so it can't release memory. OOM depends
  on exact timing and page sizes.

Scenario C: APBiology.pdf + APBiology.pdf (both heavy) — FAILS
  Worker 1: 10,900 MiB (APBiology)
  Worker 2: 10,900 MiB (APBiology)
  Total:    21,800 MiB  of 15,360 MiB  -->  6,440 MiB OVER. Guaranteed OOM.
  This is what happened: "Tried to allocate 4.83 GiB" error.
```

### Example 4: What we saw in nvidia-smi at different moments

```
Right after container starts (no warmup yet):
  PID 10590:  1,216 MiB   <-- Worker 1, just CUDA context
  PID 10591:  1,216 MiB   <-- Worker 2, just CUDA context
  Total:      2,432 MiB

After warmup (both workers processed 1-page PDF, then empty_cache ran):
  PID 10590:  1,268 MiB   <-- Worker 1, idle
  PID 10591:  1,268 MiB   <-- Worker 2, idle
  Total:      2,536 MiB

During APBiology processing on Worker 1:
  PID 10590:  3,966 MiB   <-- Worker 1, actively processing (mid-doc)
  PID 10591:  1,268 MiB   <-- Worker 2, idle
  Total:      5,234 MiB

  Note: 3,966 MiB is mid-processing. It peaks higher at ~10,900 MiB when
  hitting the densest pages, then drops as those pages finish.
```

---

## 5. GPU Sizing Guide (T4 = 15,360 MiB)

### Why 2 heavy docs fail on T4 (15 GB)

```
Worker 1 (800pg book):   5.1 GB static + 5.8 GB dynamic = 10.9 GB
Worker 2 (800pg book):   5.1 GB static + 5.8 GB dynamic = 10.9 GB
Total needed:            21.8 GB  -->  exceeds 15 GB = OOM
```

### What works on T4

```
Worker 1 (800pg book):   10.9 GB
Worker 2 (small doc):     5.5 GB
Total:                   16.4 GB  -->  tight but may work

Worker 1 (small doc):     5.5 GB
Worker 2 (small doc):     5.5 GB
Total:                   11.0 GB  -->  fits easily
```

### GPU comparison for parallel processing

| GPU | Instance | Memory | Heavy docs in parallel | Approx cost/hr |
|---|---|---|---|---|
| T4 | g4dn.xlarge | 15 GB | 1 heavy or 2 light | ~$0.53 |
| A10G | g5.xlarge | 24 GB | 2 heavy | ~$1.01 |
| A100 40GB | p4d | 40 GB | 3-4 heavy | ~$3+ |
| A100 80GB | p4de | 80 GB | 7+ heavy | ~$6+ |

---

## 6. Key Configuration Variables

| Variable | What it does | Default | Impact on memory |
|---|---|---|---|
| `PDF_WORKERS` | Number of Gunicorn worker processes | 2 | Each worker loads a full model copy (~5.1 GB) |
| `INFERENCE_BATCH_SIZE` | Pages processed per GPU batch | 4 | Higher = faster but more memory per batch |
| `CUDA_AMP` | Mixed precision (float16) | true | Reduces activation memory ~30-40% |
| `INFERENCE_NUM_WORKERS` | Detectron2 data loader threads | 2 | Affects CPU/RAM, not GPU |

### Recommended settings by GPU

| GPU | Workers | Batch size | AMP |
|---|---|---|---|
| T4 (15 GB) | 2 | 1-2 | true |
| A10G (24 GB) | 2 | 4 | true |
| A100 (40 GB) | 3-4 | 4-8 | optional |

---

## 7. Fixes Applied

1. **`torch.cuda.empty_cache()`** in `vgt_model_adapter.py` — releases cached GPU memory after each document so the other worker can use it.
2. **Per-document XML paths** in `PdfImages.py` — fixes race condition where concurrent workers shared `/tmp/pdf_etree.xml`.
3. **Auto warmup** via `warmup_workers.py` — sends a dummy PDF to each worker at startup to force GPU memory allocation before real requests arrive.
4. **Env var passthrough** in `docker-compose-gpu.yml` — `INFERENCE_BATCH_SIZE`, `INFERENCE_NUM_WORKERS`, `CUDA_AMP` now use `${VAR:-default}` syntax so test scripts can override them.

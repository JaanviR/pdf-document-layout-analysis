# Huridocs Document Isolation Changes

## Steps to Test

- Copy/push these changed files to the GPU machine
- Build with docker compose -f docker-compose-gpu.yml build
- Start with docker compose -f docker-compose-gpu.yml up -d
- Run python -m pytest src/tests/test_end_to_end.py -v inside the container
- Watch especially for test_analyze_concurrent_requests_do_not_corrupt — that's the test that proves isolation works

## The Problem

When your RAG agent sends multiple PDFs to Huridocs for processing, they queue up and process one at a time. An 800-page book takes ~4 hours on GPU, so if you send 2 books, the second one waits 4 hours before it even starts — total ~8 hours.

The root cause is that every document shares the same global file paths inside the container. All page images go to one `images/` folder, all word grids go to one `word_grids/` folder, and there's only one `test.json` annotations file. If two documents tried to process at the same time, they'd overwrite each other's files and produce garbage results.

To prevent this, the code uses a `threading.Lock()` that forces only one document through the entire pipeline at a time. This lock is a safety measure, but it's also the reason for the queue.

## The Solution

Give each document its own private workspace — a temporary directory with its own `images/`, `word_grids/`, `jsons/`, and `output/` subdirectories. This way, two documents can never interfere with each other's files, even if they process at the same time.

The workspace looks like this:

```
/tmp/pdla_a1b2c3d4/          <-- unique per document
    images/                   <-- page JPEG images for THIS document only
    word_grids/               <-- word grid pickle files for THIS document only
    jsons/                    <-- COCO annotations JSON for THIS document only
    output/                   <-- model inference results for THIS document only
```

When the document finishes processing, the entire workspace is deleted automatically.

---

## File-by-File Changes

### 1. `src/domain/PdfImages.py`

**What this file does:** Handles converting PDF pages to JPEG images and saving them to disk. The VGT model needs these images to "see" each page.

#### Change 1a: Added `images_dir` parameter to `__init__`

**Before (line 19):**
```python
def __init__(self, pdf_features, pdf_images, dpi=72):
    self.pdf_features = pdf_features
    self.pdf_images = pdf_images
    self.dpi = dpi
    self.save_images()
```

**After (line 19):**
```python
def __init__(self, pdf_features, pdf_images, dpi=72, images_dir=None):
    self.pdf_features = pdf_features
    self.pdf_images = pdf_images
    self.dpi = dpi
    self.images_dir = Path(images_dir) if images_dir else IMAGES_ROOT_PATH
    self.save_images()
```

**Why:** Previously, every PdfImages object saved images to the same global `IMAGES_ROOT_PATH` (which is `/app/images/`). Now each object can be told "save your images to THIS specific directory instead." If no directory is given, it falls back to the old global path so existing code (tests, Gradio app) still works.

#### Change 1b: `save_images()` now uses `self.images_dir` instead of the global

**Before (line 32-36):**
```python
def save_images(self):
    makedirs(IMAGES_ROOT_PATH, exist_ok=True)
    for image_index, image in enumerate(self.pdf_images):
        image_name = f"{self.pdf_features.file_name}_{image_index}.jpg"
        image.save(join(IMAGES_ROOT_PATH, image_name))
```

**After (line 32-36):**
```python
def save_images(self):
    makedirs(self.images_dir, exist_ok=True)
    for image_index, image in enumerate(self.pdf_images):
        image_name = f"{self.pdf_features.file_name}_{image_index}.jpg"
        image.save(join(self.images_dir, image_name))
```

**Why:** Images now go to the document's private workspace directory, not the shared global one. Document A's images go to `/tmp/pdla_aaa/images/`, Document B's images go to `/tmp/pdla_bbb/images/`. They can never overwrite each other.

#### Change 1c: `remove_images()` changed from static method to instance method

**Before (line 38-40):**
```python
@staticmethod
def remove_images():
    shutil.rmtree(IMAGES_ROOT_PATH)
```

**After (line 38-39):**
```python
def remove_images(self):
    shutil.rmtree(self.images_dir, ignore_errors=True)
```

**Why:** Previously, `remove_images()` was a static method that deleted the entire global `images/` directory. If Document A finished and called this while Document B was still processing, Document B's images would be deleted mid-processing, causing it to fail. Now each PdfImages object only deletes its own workspace's images directory. Also added `ignore_errors=True` so it doesn't crash if the directory was already cleaned up.

#### Change 1d: `from_pdf_path()` accepts and passes through `images_dir`

**Before (line 43):**
```python
def from_pdf_path(pdf_path, pdf_name="", xml_file_name="", dpi=72):
    ...
    return PdfImages(pdf_features, pdf_images, dpi)
```

**After (line 43):**
```python
def from_pdf_path(pdf_path, pdf_name="", xml_file_name="", dpi=72, images_dir=None):
    ...
    return PdfImages(pdf_features, pdf_images, dpi, images_dir=images_dir)
```

**Why:** This is a factory method that creates PdfImages objects. It needs to accept the `images_dir` parameter and pass it down to the constructor so the caller can control where images are saved.

#### Change 1e: `from_pdf_path_capture_xml()` accepts and passes through `images_dir`

Same change as 1d, but for the XML-capturing variant of the factory method. This method is used by the `/analyze` endpoint.

---

### 2. `src/adapters/ml/vgt/create_word_grid.py`

**What this file does:** For each page, it takes the text tokens (words and their positions) and converts them into "word grids" — structured data that tells the VGT model where each word is on the page. These are saved as `.pkl` (pickle) files.

#### Change 2a: `create_word_grid()` accepts `word_grids_dir` parameter

**Before (line 83-93):**
```python
def create_word_grid(pdf_features_list):
    makedirs(WORD_GRIDS_PATH, exist_ok=True)
    for pdf_features in pdf_features_list:
        for page in pdf_features.pages:
            image_id = f"{pdf_features.file_name}_{page.page_number - 1}"
            if exists(join(WORD_GRIDS_PATH, image_id + ".pkl")):
                continue
            grid_words_dict = get_grid_words_dict(page.tokens)
            with open(join(WORD_GRIDS_PATH, f"{image_id}.pkl"), mode="wb") as file:
                pickle.dump(grid_words_dict, file)
```

**After (line 83-93):**
```python
def create_word_grid(pdf_features_list, word_grids_dir=None):
    target_dir = word_grids_dir or WORD_GRIDS_PATH
    makedirs(target_dir, exist_ok=True)
    for pdf_features in pdf_features_list:
        for page in pdf_features.pages:
            image_id = f"{pdf_features.file_name}_{page.page_number - 1}"
            if exists(join(target_dir, image_id + ".pkl")):
                continue
            grid_words_dict = get_grid_words_dict(page.tokens)
            with open(join(target_dir, f"{image_id}.pkl"), mode="wb") as file:
                pickle.dump(grid_words_dict, file)
```

**Why:** Same idea as images. Word grid files now go to the document's private workspace (`/tmp/pdla_aaa/word_grids/`) instead of the shared global `word_grids/` directory. If no directory is given, falls back to the old global path.

#### Change 2b: `remove_word_grids()` accepts `word_grids_dir` parameter

**Before (line 96-97):**
```python
def remove_word_grids():
    shutil.rmtree(WORD_GRIDS_PATH, ignore_errors=True)
```

**After (line 96-97):**
```python
def remove_word_grids(word_grids_dir=None):
    shutil.rmtree(word_grids_dir or WORD_GRIDS_PATH, ignore_errors=True)
```

**Why:** Only deletes this document's word grids, not everyone's.

---

### 3. `src/adapters/ml/vgt/get_json_annotations.py`

**What this file does:** Creates a COCO-format JSON file that lists all the page images and their text token annotations. Detectron2 (the ML framework) requires data in this specific format to run inference.

#### Change 3a: Added `Path` import

Added `from pathlib import Path` to the imports because we now use `Path` objects for file paths.

#### Change 3b: `save_annotations_json()` accepts `json_file_path` parameter

**Before (line 9):**
```python
def save_annotations_json(annotations, width_height, images):
    ...
    JSON_TEST_FILE_PATH.write_text(json.dumps(coco_dict))
```

**After (line 9):**
```python
def save_annotations_json(annotations, width_height, images, json_file_path=None):
    target_path = json_file_path or JSON_TEST_FILE_PATH
    ...
    makedirs(target_path.parent, exist_ok=True)
    target_path.write_text(json.dumps(coco_dict))
```

**Why:** Previously, every document's annotations were written to the same global `test.json` file. If Document B started processing while Document A was running, Document B would overwrite Document A's annotations file, and Document A's model inference would use the wrong data. Now each document writes to its own JSON file in its workspace.

#### Change 3c: `get_annotations()` passes through `json_file_path`

**Before (line 59):**
```python
def get_annotations(pdf_images_list):
    makedirs(JSONS_ROOT_PATH, exist_ok=True)
    ...
    save_annotations_json(annotations, width_height, images)
```

**After (line 59):**
```python
def get_annotations(pdf_images_list, json_file_path=None):
    ...
    save_annotations_json(annotations, width_height, images, json_file_path=json_file_path)
```

**Why:** Passes the per-document JSON path down to `save_annotations_json`. Also removed the `makedirs(JSONS_ROOT_PATH)` call since `save_annotations_json` now creates the parent directory itself.

---

### 4. `src/adapters/ml/vgt/get_most_probable_pdf_segments.py`

**What this file does:** After the VGT model runs inference, it writes predictions to a JSON file. This file reads those predictions and maps them back to the original text tokens — figuring out which words belong to which predicted region (table, heading, text, etc.).

#### Change 4a: `get_vgt_predictions()` accepts `output_dir` and `json_file_path`

**Before (line 33-41):**
```python
def get_vgt_predictions(model_name):
    output_dir = f"model_output_{model_name}"
    model_output_json_path = join(str(ROOT_PATH), output_dir, "inference", "coco_instances_results.json")
    annotations = json.loads(Path(model_output_json_path).read_text())

    test_json_path = join(str(ROOT_PATH), "jsons", "test.json")
    coco_truth = json.loads(Path(test_json_path).read_text())
```

**After (line 33-43):**
```python
def get_vgt_predictions(model_name, output_dir=None, json_file_path=None):
    if output_dir:
        model_output_json_path = join(output_dir, "inference", "coco_instances_results.json")
    else:
        default_output_dir = f"model_output_{model_name}"
        model_output_json_path = join(str(ROOT_PATH), default_output_dir, "inference", "coco_instances_results.json")
    annotations = json.loads(Path(model_output_json_path).read_text())

    test_json_path = json_file_path or join(str(ROOT_PATH), "jsons", "test.json")
    coco_truth = json.loads(Path(test_json_path).read_text())
```

**Why:** Two paths were hardcoded to global locations:
1. The model output (`coco_instances_results.json`) — where the model writes its predictions
2. The COCO truth (`test.json`) — where the image/annotation list is stored

Both now point to the document's private workspace when provided.

#### Change 4b: `get_most_probable_pdf_segments()` passes through both parameters

**Before (line 126):**
```python
def get_most_probable_pdf_segments(model_name, pdf_images_list, save_output=False):
    vgt_predictions_dict = get_vgt_predictions(model_name)
```

**After (line 126):**
```python
def get_most_probable_pdf_segments(model_name, pdf_images_list, save_output=False, output_dir=None, json_file_path=None):
    vgt_predictions_dict = get_vgt_predictions(model_name, output_dir=output_dir, json_file_path=json_file_path)
```

**Why:** Just passes the per-document paths through to the function that actually reads the files.

---

### 5. `src/adapters/ml/vgt_model_adapter.py`

**What this file does:** This is the main orchestrator for VGT model inference. It coordinates all the steps: create word grids → create annotations → register dataset → run model → read predictions → clean up.

This file had the most significant changes.

#### Change 5a: New imports added

Added `shutil`, `tempfile`, `uuid`, and `Path` imports. These are needed to create and clean up per-document workspace directories.

#### Change 5b: Removed `_register_data()` method and rewrote `predict_document_layout()`

**Before (line 84-107):**
```python
class VGTModelAdapter(MLModelService):

    def _register_data(self):
        try:
            DatasetCatalog.remove("predict_data")
        except KeyError:
            pass
        register_coco_instances("predict_data", {}, JSON_TEST_FILE_PATH, IMAGES_ROOT_PATH)

    def predict_document_layout(self, pdf_images):
        create_word_grid([img.pdf_features for img in pdf_images])
        get_annotations(pdf_images)
        self._register_data()
        with suppress_logs():
            VGTTrainer.test(configuration, model)
        predicted_segments = get_most_probable_pdf_segments("doclaynet", pdf_images, False)
        PdfImages.remove_images()
        remove_word_grids()
        return get_reading_orders(pdf_images, predicted_segments)
```

**After (line 87-139):**
```python
class VGTModelAdapter(MLModelService):

    def predict_document_layout(self, pdf_images):
        workspace_id = uuid.uuid4().hex[:8]
        workspace_dir = Path(pdf_images[0].images_dir).parent
        word_grids_dir = str(workspace_dir / "word_grids")
        json_file_path = workspace_dir / "jsons" / "test.json"
        output_dir = str(workspace_dir / "output")
        dataset_name = f"predict_data_{workspace_id}"

        try:
            create_word_grid(
                [img.pdf_features for img in pdf_images],
                word_grids_dir=word_grids_dir,
            )
            get_annotations(pdf_images, json_file_path=json_file_path)

            try:
                DatasetCatalog.remove(dataset_name)
            except KeyError:
                pass
            register_coco_instances(
                dataset_name, {}, str(json_file_path), str(pdf_images[0].images_dir),
            )

            doc_cfg = configuration.clone()
            doc_cfg.defrost()
            doc_cfg.OUTPUT_DIR = output_dir
            doc_cfg.DATASETS.TEST = (dataset_name,)
            doc_cfg.freeze()

            with suppress_logs():
                VGTTrainer.test(doc_cfg, model)

            predicted_segments = get_most_probable_pdf_segments(
                "doclaynet", pdf_images, False,
                output_dir=output_dir,
                json_file_path=str(json_file_path),
            )

            return get_reading_orders(pdf_images, predicted_segments)
        finally:
            try:
                DatasetCatalog.remove(dataset_name)
            except KeyError:
                pass
            for pdf_img in pdf_images:
                pdf_img.remove_images()
            remove_word_grids(word_grids_dir=word_grids_dir)
            shutil.rmtree(workspace_dir, ignore_errors=True)
```

**Why — step by step:**

1. **`workspace_id = uuid.uuid4().hex[:8]`** — generates a unique 8-character ID like `a1b2c3d4`. This ensures no two documents can collide, even if they start at the exact same time.

2. **`workspace_dir = Path(pdf_images[0].images_dir).parent`** — the workspace directory was already created by `pdf_analysis_service_adapter.py` (see change 6 below). The images are already saved in `workspace/images/`, so we go up one level to get the workspace root.

3. **`word_grids_dir`, `json_file_path`, `output_dir`** — all derived from the workspace. Everything stays in one place.

4. **`dataset_name = f"predict_data_{workspace_id}"`** — previously, every document was registered as `"predict_data"` in Detectron2's dataset catalog. If two documents tried to register at the same time, the second would overwrite the first's registration. Now each document gets a unique name like `"predict_data_a1b2c3d4"`.

5. **`doc_cfg = configuration.clone()` → `defrost()` → modify → `freeze()`** — the Detectron2 configuration (which includes the model output directory and dataset name) is created once at startup and frozen. We can't modify it directly. So we create a copy for this document, unfreeze it, set the per-document output directory and dataset name, then freeze it again. The actual model weights (`model`) are shared — only the config changes.

6. **`finally:` block** — no matter what happens (success or error), we always clean up: unregister the dataset name, delete the images, delete the word grids, and delete the entire workspace directory. This prevents temp files from building up on disk.

#### What stayed the same

The model loading at the top of the file (lines 81-84) is unchanged:
```python
with suppress_logs():
    configuration = get_model_configuration()
    model = VGTTrainer.build_model(configuration)
    DetectionCheckpointer(model, ...).resume_or_load(...)
```
The model is still loaded once at startup and reused for every document. Only the configuration and file paths change per document.

---

### 6. `src/adapters/infrastructure/pdf_analysis_service_adapter.py`

**What this file does:** This is the entry point for all PDF analysis. When a request comes in, this file saves the PDF to disk, creates PdfImages objects, calls the ML model, and returns the results.

#### Change 6a: New imports and `_create_workspace()` helper

**Added:**
```python
import tempfile
from pathlib import Path

def _create_workspace() -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="pdla_"))
    (workspace / "images").mkdir()
    return workspace
```

**Why:** `_create_workspace()` creates a unique temporary directory like `/tmp/pdla_a1b2c3d4/` and creates an `images/` subdirectory inside it. The `tempfile.mkdtemp()` function guarantees the directory name is unique. The `pdla_` prefix makes it easy to identify these directories if you need to debug.

#### Change 6b: `analyze_pdf_layout()` creates workspace and passes `images_dir`

**Before (line 32):**
```python
pdf_images_list = [PdfImages.from_pdf_path(pdf_path, "", xml_filename)]
```

**After (line 39-42):**
```python
workspace = _create_workspace()
images_dir = workspace / "images"
pdf_images_list = [PdfImages.from_pdf_path(pdf_path, "", xml_filename, images_dir=images_dir)]
```

**Why:** Creates a private workspace for this document and tells PdfImages to save its page images there instead of the global directory. The same `images_dir` is passed to the 200 DPI image creation (for table/formula parsing) so those images also go to the workspace.

#### Change 6c: Same change applied to all four `analyze_*` methods

The same workspace creation pattern was applied to:
- `analyze_pdf_layout()` — normal VGT analysis
- `analyze_pdf_layout_fast()` — fast LightGBM analysis  
- `analyze_pdf_layout_with_xml()` — VGT analysis with XML output
- `analyze_pdf_layout_fast_with_xml()` — fast analysis with XML output

All four now create their own workspace and pass `images_dir` to PdfImages.

#### What stayed the same

The `_predict_lock` is still there. Right now it still serializes all document processing. But with file isolation in place, the lock is only needed for GPU access (the model object is shared). In the next phase, we can either:
- Remove it and add multiple gunicorn workers (each gets its own model)
- Or run multiple containers (each fully independent)

---

### 7. `src/adapters/ml/fast_trainer_adapter.py`

**No changes needed.** This file already calls `pdf_images_obj.remove_images()` as an instance method call. Since `remove_images()` is now an instance method that uses `self.images_dir`, it will correctly clean up only this document's images from its workspace. The fast adapter doesn't use word grids or COCO annotations, so no other changes are needed.

---

## What This Enables (Next Steps)

With file isolation in place, we can now safely:

1. **Run multiple container replicas** behind a load balancer — each container processes a different document simultaneously with zero file conflicts.

2. **Increase the inference batch size** — the VGT model currently processes pages one at a time (batch size = 1). We can batch 4-8 pages together for the same document, making much better use of the GPU.

3. **Enable GPU optimizations** (cudnn.benchmark, AMP) — these speed up the model inference without any quality loss.

4. **Narrow or remove the `_predict_lock`** — with files isolated, the lock only needs to protect the GPU model object, not the entire pipeline. Preprocessing (PDF → images → word grids) can happen in parallel with another document's model inference.

## Backwards Compatibility

All new parameters default to `None`, which falls back to the old global paths. This means:
- The Gradio GUI app still works without changes
- Existing tests still work without changes
- Any code that calls these functions without the new parameters behaves exactly as before

# Session Notes — 2026-05-30

## What happened

### Phase 1 — TF-IDF baseline archived

`~/Documents/claude_creations/2026-05-30_091736_doc-classifier/`

- `doc_classifier.py` moved from `~/Downloads/` and documented
- `~/doc-clf-env` created; deps: `scikit-learn joblib pdfplumber python-docx striprtf`
- Training attempted against `2026-05-28_militia-copy/` — killed by user before completion
  - Training data confirmed: 69 labelled sub-folders, 1887 PDFs, majority images (skipped)
  - `train_stderr.txt` captures extraction errors from that run

### Phase 2 — GPU embedding version archived

`~/Documents/claude_creations/2026-05-30_092836_doc-classifier-gpu/`

- `doc_classifier_gpu.py` moved from `~/Downloads/` and documented
- Architecture: `extract_text()` → overlapping 4000-char chunks → bge-m3 on GPU → mean-pool → LogisticRegression

### Phase 3 — Blackwell GPU environment verified

Created `~/doc-clf-gpu-env` with cu130 nightly PyTorch:

```
torch: 2.13.0.dev20260521+cu130
GPU 0: NVIDIA GeForce RTX 5080  capability=(12, 0)  ← Blackwell, sm_120 confirmed
GPU 1: NVIDIA GeForce RTX 3080  capability=(8, 6)
```

Full dep stack installed and import-tested:
- `torch 2.13.0.dev+cu130` — sm_120 kernels present
- `sentence-transformers`, `easyocr`, `pymupdf`
- `scikit-learn`, `joblib`, `pdfplumber`, `python-docx`, `striprtf`

---

## Script evolution (chronological)

| Version | Commit | Change |
|---------|--------|--------|
| v1 | f82cd48 | Basic pipeline: extract all → load GPU model → embed all. GPU idle during extraction. |
| v2 | d138ab5 | Per-file verbose output (flush trick). Interleaved per-class extract→embed so GPU fires after each class, not only at the end. |
| v3 | d138ab5 | EasyOCR + PyMuPDF GPU OCR fallback for scanned PDFs and `.jpg`/`.png` files. `--workers` flag (default 14). |
| v4 | d138ab5 | `ProcessPoolExecutor` for true parallel CPU extraction (separate interpreter per worker, no GIL). **Crashed** — see Incidents. |
| v5 | 8c0c8a6 | Wrong fix: switched to `ThreadPoolExecutor`. pdfminer/pdfplumber/pypdf are pure Python; threads serialize at the GIL. No actual parallelism. |
| v6 | e2f2d30 | Restored `ProcessPoolExecutor` + two-level `BrokenProcessPool` guard. Worker segfaults trigger sequential fallback. Insufficient — see Incidents (mass crashes). |
| v7 | 7c4380d | `rich` TUI: per-class Progress bars (CPU/fallback/OCR phases), GPU stat lines, Panel headers, summary Table, predict Table output. |
| v8 | b0ea134 | pdfplumber → PyMuPDF (fitz) for PDF text extraction — eliminates Pillow segfaults entirely. `ProcessPoolExecutor` → `multiprocessing.Pool(maxtasksperchild=1)` — per-file crash isolation + 60s timeout. |
| v9 | f75eae7 | Two bug fixes: (1) `console.print(..., stderr=True)` invalid — rich has no `stderr` param, causing exception handlers to re-raise instead of swallow. Fixed to `print(..., file=sys.stderr)`. (2) macOS resource fork files (`._filename`) passed to EasyOCR as images → crash. Fixed by filtering `f.name.startswith("._")` in all file lists. |
| v10 | d488178 | **Current.** Automatic GPU fallback: `_pick_device(min_free_mb)` probes each visible CUDA device for free VRAM, returns first with headroom, falls back to CPU. `get_encoder()` and `get_ocr_reader()` use it — no more manual `CUDA_VISIBLE_DEVICES`. Callers derive `device` from `str(encoder.device)` after load. |

---

## Incidents

### BrokenProcessPool crash — 2026-05-30 (training run 1)

**Class:** Liberated_manuals (706 files)

**Error:**
```
concurrent.futures.process.BrokenProcessPool: A process in the process pool was
terminated abruptly while the future was running or pending.
```

**Root cause (corrected):** NOT OOM — worlock had 101 GB RAM free at the time.
Actual cause: Pillow's C extensions (`_imaging.so`) are invoked when pdfplumber encounters
a PDF with embedded images. A corrupt image in a Liberated_manuals PDF caused a segfault
inside the C code, killing that worker process and poisoning all pending futures in the pool.

**Wrong intermediate fix (v5):** Switched to `ThreadPoolExecutor`. This appeared to solve
the crash but sacrificed all parallelism — pdfminer, pdfplumber, and pypdf are pure Python;
the GIL serializes threads, so 14 "workers" execute one at a time.

**Diagnosis:** Confirmed by checking: `pdfminer`, `pdfplumber`, `pypdf` → zero `.so` files
(pure Python). Pillow → 8 `.so` C extensions. 101 GB RAM free → OOM ruled out.

**Partial fix (v6):** Restored `ProcessPoolExecutor` with two-level exception handling.
The pool survived individual crashes, but crashes were too frequent to be effective —
see next incident.

---

### Mass worker crashes — 2026-05-30 (training run 3)

**Class:** Liberated_manuals (697 PDFs)

**Symptom:** Hundreds of `worker error — retrying sequentially` lines in console. The
class that should benefit most from parallelism was processing almost entirely sequentially.

**Root cause (deep):** Not one corrupt PDF — the problem is structural. pdfplumber invokes
`pdfminer.image.PDFImageInterpreter` → `PIL.Image.open()` → `_imaging.so` for **every
PDF containing embedded images**. Sampling 50 PDFs from Liberated_manuals: **49/50
(98%) contain embedded images**. With 697 PDFs and 14 workers, Pillow is invoked ~683
times. Any malformed image in any of those files triggers a segfault.

**Investigation commands:**
```bash
# Count image-bearing PDFs
python3 -c "
import pdfplumber; from pathlib import Path
lm = Path('.../Liberated_manuals')
has_img = sum(1 for p in list(lm.glob('*.pdf'))[:50]
              if any(pg.images for pg in pdfplumber.open(p).pages[:3]))
print(f'{has_img}/50 PDFs have embedded images')  # → 49/50
"

# pdfminer.image imports PIL on line 154:
#   from PIL import Image, ImageChops  # triggers _imaging.so
```

**Fix (v8 — b0ea134):**
1. **Replace pdfplumber with PyMuPDF (fitz)** for PDF text extraction.
   `fitz.page.get_text()` handles embedded images internally via MuPDF's own C layer —
   Pillow is never invoked. Smoke test on three largest PDFs (66-69 MB, 312 pages):
   ```
   tc3-97-61.pdf:  720,290 chars in 0.31s  ✓
   fm3-97-61.pdf:  668,254 chars in 0.29s  ✓
   doplaw-v2.pdf: 1,043,753 chars in 0.50s ✓
   ```
2. **Replace ProcessPoolExecutor with `multiprocessing.Pool(maxtasksperchild=1)`.**
   Each worker handles exactly 1 file then exits. Any crash is isolated to that one file;
   the pool spawns a fresh worker for the next. Per-task 60s timeout also catches hangs.

---

## New tools and documents added this session

| File | Description |
|------|-------------|
| `sort_docs.py` | Bulk classifier sorter — classify 26k+ files, copy/symlink/report into class folders |
| `wizard.py` | Interactive training-to-inference wizard (7-step, rich TUI) |
| `MILITIA_MODEL.md` | Model reference: lay + expert explanation of militia.joblib internals |
| `HOWTO.md` | Step-by-step operational how-to guide |
| `hypersphere/HYPERSPHERE.md` | Unit hypersphere concept: lay + expert + 10 visualizations |
| `hypersphere/hyper[1-5].png/svg` | matplotlib: normalization, sphere patches, cosine similarity, decision boundary, pipeline |
| `hypersphere/dot[1-5].png/svg` | Graphviz: normalization math, TF-IDF vs embedding, inference flow, class geometry, train vs infer |

---

## Current state (2026-05-30)

- **Training run 4** (v8 code, PyMuPDF + Pool + rich TUI) in progress — pid 97589, GPU 0 at capacity
- **`militia.joblib`** — produced by run 3 (v6 code), 64 classes, 522 KB, available
- **`militia_v8.joblib`** — not yet produced (run 4 still running)
- **sort_docs.py report run** in progress — pid 105699, GPU 1 (RTX 3080, 3.2 GB in use)
  - Target: 26,577 PDFs from USAFA corpus
  - Mode: report (no file operations), threshold 60%
  - bge-m3 loaded on GPU 1 automatically via `_pick_device()`
- bge-m3 cached at `~/.cache/huggingface/`; EasyOCR at `~/.EasyOCR/`

---

## Environments

| venv | Purpose | PyTorch |
|------|---------|---------|
| `~/doc-clf-env` | TF-IDF baseline (CPU only) | none |
| `~/doc-clf-gpu-env` | GPU embeddings (Blackwell) | 2.13.0.dev+cu130 |

---

## Resume command

```bash
source ~/doc-clf-gpu-env/bin/activate
cd ~/Documents/claude_creations/2026-05-30_092836_doc-classifier-gpu/

DISPLAY=:0 xterm -title "doc_classifier_gpu — training" -fa 'Monospace' -fs 11 -geometry 120x50 -e bash -c '
source ~/doc-clf-gpu-env/bin/activate
cd ~/Documents/claude_creations/2026-05-30_092836_doc-classifier-gpu
CUDA_VISIBLE_DEVICES=0 python doc_classifier_gpu.py train \
  ~/Documents/claude_creations/2026-05-28_militia-copy -m militia.joblib
echo; echo "--- done (exit $?) --- press Enter to close ---"; read
' &
```

---

## Lessons learned

### 1. pdfplumber + embedded images → Pillow segfaults (root cause of all crashes)
`pdfplumber.page.extract_text()` internally calls `pdfminer.image` → `PIL.Image.open()`
→ `_imaging.so` C extension for *every* PDF with embedded images. 98% of Liberated_manuals
PDFs have embedded images. Any corrupt image data → SIGSEGV → worker process killed.
**Fix:** Replace pdfplumber with PyMuPDF (`fitz`). MuPDF handles images internally
in C without exposing raw image bytes to Pillow.

### 2. ThreadPoolExecutor does not parallelize pure-Python CPU work
pdfminer, pdfplumber, and pypdf have **zero C extensions** — they are pure Python.
The GIL serializes all threads. 14 `ThreadPoolExecutor` workers appear to run in
parallel but execute one at a time. `ProcessPoolExecutor` (separate interpreters)
is required for true CPU parallelism on pure-Python code.

### 3. ProcessPoolExecutor BrokenProcessPool — isolation, not just resilience
The first BrokenProcessPool fix (two-level guard) caught crashes but didn't prevent
them. Each crash still serialized remaining files to sequential fallback.
`Pool(maxtasksperchild=1)` is the right fix: each worker handles 1 file then exits,
so a crash affects only that one file. The pool spawns a fresh worker for the next.

### 4. `console.print(..., stderr=True)` is wrong — rich has no `stderr` param
`rich.Console.print()` does not accept `stderr=True`. Setting it causes `TypeError`
inside the `except` block, which re-raises instead of swallowing the original exception.
Use `print(..., file=sys.stderr)` for error output to stderr.

### 5. macOS resource fork files (`._filename`) must be filtered
When a macOS system copies files to a Linux filesystem, it creates binary metadata
sidecars named `._original_filename`. These appear to have image extensions (`.jpg`,
`.png`) but are binary Apple Double format — EasyOCR crashes on them. Always filter
`f.name.startswith("._")` before sending files to any extraction or OCR pipeline.

### 6. GPU memory awareness — probe before loading, not after OOM
Loading bge-m3 onto a GPU that has only 59 MB free raises `torch.OutOfMemoryError`
with no recovery. The correct pattern is to probe free VRAM with
`torch.cuda.mem_get_info(i)` before attempting the load, select the best available
device, and only use the OOM try-except as a belt-and-suspenders fallback.

### 7. Training in a visible xterm is essential for operator awareness
Running training as a background subprocess hides all output. The operator cannot
see errors, progress, or GPU activity. Always launch long-running GPU jobs in a
separate `xterm` window so the operator can monitor and intervene.

### 8. Two GPUs can run independent workloads simultaneously
Training on GPU 0 + classification on GPU 1 worked cleanly once `_pick_device()`
automatically selected GPU 1 based on free VRAM. No coordination needed — PyTorch
and sentence-transformers handle device isolation correctly.

---

## Environments

| venv | Purpose | PyTorch |
|------|---------|---------|
| `~/doc-clf-env` | TF-IDF baseline (CPU only) | none |
| `~/doc-clf-gpu-env` | GPU embeddings (Blackwell) | 2.13.0.dev+cu130 |

---

## Resume command

```bash
DISPLAY=:0 xterm -title "doc_classifier_gpu — training" -fa 'Monospace' -fs 11 -geometry 130x55 -e bash -c '
source ~/doc-clf-gpu-env/bin/activate
cd ~/Documents/claude_creations/2026-05-30_092836_doc-classifier-gpu
python doc_classifier_gpu.py train \
  ~/Documents/claude_creations/2026-05-28_militia-copy -m militia_v8.joblib
echo; echo "--- done (exit $?) --- press Enter to close ---"; read
' &
```

## Sort run command (after model is ready)

```bash
DISPLAY=:0 xterm -title "sort_docs" -fa 'Monospace' -fs 11 -geometry 130x55 -e bash -c '
source ~/doc-clf-gpu-env/bin/activate
cd ~/Documents/claude_creations/2026-05-30_092836_doc-classifier-gpu
python sort_docs.py /path/to/target_dir \
  --model militia.joblib --output ./sorted --mode copy --threshold 0.60
echo; echo "--- done --- press Enter to close ---"; read
' &
```

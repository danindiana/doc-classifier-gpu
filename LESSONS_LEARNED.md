# Lessons Learned — doc_classifier_gpu

Operational lessons from building and debugging the GPU embedding document
classifier pipeline on worlock (RTX 5080 + RTX 3080, Ubuntu, Python 3.10).

---

## 1. pdfplumber + embedded images → Pillow segfaults

**What happened:** `pdfplumber.page.extract_text()` internally calls
`pdfminer.image.PDFImageInterpreter` → `PIL.Image.open()` → `_imaging.so` C extension
for every PDF containing embedded images. 98% of the Liberated_manuals class (697 PDFs)
have embedded images. Any corrupt image data → SIGSEGV → worker process killed.

**How we found it:** Sampled 50 PDFs — 49/50 had embedded images. Checked
`pdfminer.image` source — PIL import on line 154. dmesg showed no OOM.

**Fix:** Replace pdfplumber with PyMuPDF (`fitz`). `fitz.page.get_text()` handles
embedded images internally via MuPDF's own C layer — Pillow is never invoked.

**Rule:** For any PDF pipeline running in parallel workers: audit whether the PDF
library calls Pillow under the hood. If it does, use PyMuPDF instead.

---

## 2. ThreadPoolExecutor does not parallelize pure-Python CPU work

**What happened:** Switched from `ProcessPoolExecutor` to `ThreadPoolExecutor` to
avoid BrokenProcessPool. Appeared to fix the crash but silently serialized all work.
pdfminer, pdfplumber, and pypdf have zero C extensions — the GIL serializes all threads.

**How we found it:** `mpstat` showed only 1 core busy during "parallel" extraction
despite 14 workers. Confirmed: `pdfminer`, `pdfplumber`, `pypdf` → zero `.so` files.

**Fix:** `ProcessPoolExecutor` (separate interpreters) for true CPU parallelism.

**Rule:** Threads parallelize I/O-bound and C-extension work (Pillow, numpy). For
pure-Python CPU work (PDF parsing, text processing), use processes.

---

## 3. ProcessPoolExecutor crash isolation — use maxtasksperchild=1

**What happened:** The first BrokenProcessPool fix (two-level try/except guard) caught
crashes but didn't prevent them. Each crash still serialized all remaining futures.

**Fix:** `multiprocessing.Pool(maxtasksperchild=1)` — each worker handles exactly 1
task then exits cleanly. A crash affects only that file. Pool spawns a fresh worker
for the next. Per-task 60s timeout also catches infinite loops on malformed files.

**Rule:** For parallel processing of untrusted files with C extension libraries, always
use `maxtasksperchild=1` (or `max_tasks_per_child` in Python 3.12+). The overhead is
negligible on Linux (fork is fast); the isolation is invaluable.

---

## 4. `console.print(..., stderr=True)` is not a valid rich API call

**What happened:** Error handling in `extract_text()` used `console.print(..., stderr=True)`.
Rich's `Console.print()` has no `stderr` parameter. This caused `TypeError` inside the
`except` block, re-raising instead of swallowing the original exception — making every
OCR error fatal.

**Fix:** Use `print(f"...", file=sys.stderr)` for stderr output, or create a separate
`Console(stderr=True)` instance at module level.

**Rule:** Check rich API docs before assuming `print()` keyword args carry over.
The `stderr` param belongs to the `Console()` constructor, not `console.print()`.

---

## 5. macOS resource fork files (`._filename`) crash EasyOCR

**What happened:** When macOS copies files to a Linux filesystem it creates binary
metadata sidecars named `._original_filename` (Apple Double format). These appear
to have image extensions (`.jpg`, `.png`) but are binary metadata — not real images.
EasyOCR attempted to read them and crashed.

**Fix:** Filter `f.name.startswith("._")` in every file listing before passing to
any extraction or OCR pipeline.

**Rule:** On any Linux system that has ever received files from macOS (USB, SMB share,
rsync), always filter `._` files. They are invisible in Finder but present on the
filesystem. Add the filter once at the file-listing stage, not at every consumer.

---

## 6. Probe GPU free VRAM before loading models, not after OOM

**What happened:** `sort_docs.py` crashed with `torch.OutOfMemoryError` at model load
because training was occupying 15 GB on GPU 0 with only 59 MB free.

**Fix:** `_pick_device(min_free_mb=2500)` probes `torch.cuda.mem_get_info(i)` for each
visible GPU, returns the first with sufficient headroom, falls back to CPU. OOM
try-except added as belt-and-suspenders.

**Rule:** Never assume `cuda` means GPU 0 is available. When running multiple GPU
workloads, probe before loading. `torch.cuda.mem_get_info(i)` returns `(free, total)`
in bytes — fast, no model load required.

---

## 7. Run training in a visible xterm, not as a background subprocess

**What happened:** First training attempts used `run_in_background=True`. All output
was hidden — operator couldn't see errors, progress, or whether the GPU was active.

**Fix:** Always launch training in a separate `xterm` window:
```bash
DISPLAY=:0 xterm -title "training" -fa 'Monospace' -fs 11 -geometry 130x55 \
  -e bash -c 'source ~/doc-clf-gpu-env/bin/activate; python doc_classifier_gpu.py train ...; read' &
```

**Rule:** Any process the operator needs to monitor (ML training, large file jobs)
belongs in an xterm, not a background subprocess. Background is for fire-and-forget.

---

## 8. Two GPUs can run independent workloads simultaneously

**What happened:** Training on GPU 0 occupied 15 GB. Assumed we had to wait.

**Fix:** `_pick_device()` selected GPU 1 (RTX 3080, 9 GB free) automatically.
Training continued on GPU 0 while classification ran on GPU 1 with no coordination
needed. PyTorch device isolation handles it correctly.

**Rule:** With multiple GPUs and automatic device selection, independent workloads
can run in parallel without manual `CUDA_VISIBLE_DEVICES` coordination.

---

## 9. GPU OCR on bulk collections — page limits are essential

**What happened:** `sort_docs.py` sent 1,664 image-only PDFs to EasyOCR at 2× render
scale, processing every page. Many were 100–300 page scanned manuals. Rate: ~3 min/file
→ estimated 91 hours to complete the OCR phase for the USAFA corpus.

**Fix:** Added `--skip-ocr` flag (image-only files go to `_review/`, run completes
in ~14 min) and `--max-ocr-pages N` flag (OCR only the first N pages).

**Rule:** For bulk classification, OCR is not worth the time if the goal is sorting
rather than full text indexing. Use `--skip-ocr` for a fast first pass, then OCR
selectively on the `_review/` subset if needed. For partial OCR, 3–5 pages is usually
enough to determine document class from section headers and opening paragraphs.

**Added parameters:**
- `sort_docs.py --skip-ocr` — fastest, image files → `_review/`
- `sort_docs.py --max-ocr-pages 3` — sample first 3 pages, ~10× faster than full OCR
- `doc_classifier_gpu.py` `_ocr_pdf(max_pages=N)` — same for training

---

## 10. MuPDF stderr warnings are not errors

**What happened:** The xterm filled with `MuPDF error: syntax error: no XObject
subtype specified` and `format error: object is not a stream`. These look alarming
but are not failures — MuPDF logs these for malformed/corrupt PDFs but still extracts
whatever text is available. The process continued correctly.

**Rule:** MuPDF's stderr warnings are informational. A genuine fitz failure raises
a Python exception caught by the `try/except` block in `extract_text()`. If the
progress bar is advancing, the job is running — ignore the MuPDF noise.

---

## Summary table

| # | Lesson | Fix |
|---|--------|-----|
| 1 | pdfplumber triggers Pillow on image-bearing PDFs | Use PyMuPDF (fitz) |
| 2 | ThreadPoolExecutor serializes pure-Python CPU work | Use ProcessPoolExecutor |
| 3 | BrokenProcessPool poisons all pending futures | `maxtasksperchild=1` |
| 4 | `console.print(stderr=True)` is not valid rich | `print(..., file=sys.stderr)` |
| 5 | macOS `._` files crash OCR | Filter `f.name.startswith("._")` |
| 6 | OOM at model load when GPU is busy | `_pick_device()` — probe before loading |
| 7 | Background training hides errors | Always use xterm |
| 8 | Assumed one GPU at a time | Auto device selection, both GPUs simultaneously |
| 9 | Full OCR on bulk corpus = 91 hours | `--skip-ocr` or `--max-ocr-pages 3` |
| 10 | MuPDF stderr output looks like failures | Informational only — check progress bar |

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

## 11. `imap_unordered` + worker crash = BrokenPipeError cascade

**What happened:** `sort_docs.py` used `pool.imap_unordered()`. When one worker crashed
(SIGSEGV from a corrupt PDF in fitz), Python raised `WorkerLostError` through the iterator.
The `except Exception` block caught it and fell out of the `with Pool` context manager.
`pool.__exit__` always calls `pool.terminate()`, which closed all pipe connections. Every
remaining worker that tried to return a result got `BrokenPipeError`, flooding stderr with
hundreds of cascading tracebacks.

**Root cause:** `pool.__exit__()` calls `pool.terminate()` unconditionally — there is no
"exit cleanly" mode. Any early exit from the `with Pool` block (whether via exception or
normal control flow) terminates all workers immediately.

**Fix:** Wrap the `with Pool` block in a `while files_todo` restart loop. Track
`done_paths` (set of path strings that returned results) and `fail_counts` (how many
crashes each path has survived). After each pool exit, restart with only the unfinished
files. Blacklist files that fail `MAX_FAILS=2` times.

```python
while files_todo and not _shutdown.is_set():
    progressed = 0
    with Pool(...) as pool:
        try:
            for path_str, text in pool.imap_unordered(_extract_cpu, files_todo, chunksize=1):
                done_paths.add(path_str)
                progressed += 1
                ...
        except Exception as exc:
            console.print(f"⚠ Worker crash: {exc}")
    files_todo = [p for p in files_todo if p not in done_paths]
    # (blacklist logic for persistent crashers)
```

**Rule:** Never use `imap_unordered` inside a bare `with Pool` if a worker crash should be
survivable. Use a restart loop with done-set tracking, or use `apply_async` with
`error_callback=` which survives individual crashes without killing the pool.

---

---

## 12. xterm has no clipboard copy — use alacritty for operator-visible jobs

**What happened:** sort_docs.py was launched in xterm. When errors appeared, the
operator couldn't copy the traceback — xterm uses X11 primary selection
(mouse-highlight → middle-click paste only), not the desktop clipboard. Errors had
to be re-typed by hand.

**Fix:** Relaunch in alacritty (`/usr/local/bin/alacritty`). Mouse-select any text
→ auto-copied to clipboard. Ctrl+Shift+C also works. Paste with Ctrl+Shift+V or
middle-click. Other copyable alternatives: xfce4-terminal, gnome-terminal, kitty.

**Rule:** Any job where the operator might need to copy error output belongs in
alacritty or another modern terminal. Reserve xterm for headless/scriptable use
where no human needs to read or copy the output.

**Launch pattern:**
```bash
DISPLAY=:0 alacritty --title "job name" -e bash -c '
  source ~/venv/bin/activate
  python my_script.py ...
  echo "--- done --- press Enter ---"; read
' &
```

---

## 13. bge-m3 on RTX 5080: model uses ~14 GB, encode-batch=512 OOM

**What happened:** bge-m3 loaded onto RTX 5080 (16 GB total). Model weights +
overhead consumed ~14.7 GB, leaving ~1.24 GB free. The embed background thread
attempted `encoder.encode(batch_size=512)` on 512 text chunks → needed 2.94 GB
for activations → `torch.OutOfMemoryError`. The embed thread crashed silently;
extraction kept running but results queued with no consumer.

**Root cause:** encode-batch=512 means 512 chunks × ~1000 tokens each go through
the transformer in one forward pass. XLM-Roberta attention scales with
batch×seq_len. With only 1.24 GB headroom, any batch >~100 chunks OOMs.

**Fix:**
1. Relaunch with `--encode-batch 32` (fits in ~1.2 GB; 32 chunks × 1000 tokens).
2. Added OOM recovery loop in `embed_batch()` (`sort_docs.py`): on
   `OutOfMemoryError`, halves `encode_batch` (min 4), calls `torch.cuda.empty_cache()`,
   and retries. Self-heals without crashing the embed thread.

**Rule:** For bge-m3 on a 16 GB GPU, default `--encode-batch` to 32. Add the OOM
recovery loop as standard boilerplate in any `encoder.encode()` call site.

```python
while True:
    try:
        vecs = encoder.encode(chunks, batch_size=encode_batch, ...)
        break
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" not in str(e).lower() or encode_batch <= 4:
            raise
        torch.cuda.empty_cache()
        encode_batch = max(4, encode_batch // 2)
```

---

---

## 14. Dual-GPU subprocess isolation — confirmed working

**What happened:** All previous dual-GPU attempts failed with glibc heap corruption
when `load_encoders()` called `SentenceTransformer(model, device="cuda:0")` then
`SentenceTransformer(model, device="cuda:1")` in the same process. The second load
corrupted the HuggingFace tokenizer Rust FFI heap, crashing immediately.

**Root cause:** The Rust FFI layer (`tokenizers` library) maintains global state that
is not safe to initialize twice in one Python process. This is unrelated to VRAM,
threading, or CUDA context switching.

**Fix:** Load each encoder in its own subprocess via `mp.get_context('spawn')`.
`spawn` gives each subprocess a fresh Python interpreter with its own heap — the
two `SentenceTransformer` instances never share any state:

```python
ctx = mp.get_context('spawn')
in_qs  = [ctx.Queue(), ctx.Queue()]
out_qs = [ctx.Queue(), ctx.Queue()]
for i in range(2):
    p = ctx.Process(target=_embed_proc_fn,
                    args=(in_qs[i], out_qs[i], model, clf_path, ..., f"cuda:{i}"))
    p.start()
```

The main process loads only the tiny sklearn classifier (522 KB, no GPU). Batches
are round-robined between the two subprocess input queues. A collector thread in the
main process merges result rows from both output queues.

**Verified result (2026-05-30):**
- RTX 5080 (GPU 0): 88% utilization, ~5,200 MiB
- RTX 3080 (GPU 1): 100% utilization, ~5,100 MiB
- No heap corruption, no crash
- ~2× embedding throughput vs single-GPU

**Key `spawn` vs `fork` distinction:** CUDA's runtime documentation explicitly
prohibits `fork` after CUDA initialization. `spawn` avoids this by starting a fresh
interpreter — the child never inherits the parent's CUDA context.

**Rule:** To run multiple `SentenceTransformer` (or any HuggingFace model) instances
concurrently: use `mp.get_context('spawn')` and spawn one process per model. Never
load two instances in the same process or via `fork`.

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
| 7 | Background training hides errors | Use alacritty / xterm (visible terminal) |
| 8 | Assumed one GPU at a time | Auto device selection, both GPUs simultaneously |
| 9 | Full OCR on bulk corpus = 91 hours | `--skip-ocr` or `--max-ocr-pages 3` |
| 10 | MuPDF stderr output looks like failures | Informational only — check progress bar |
| 11 | `imap_unordered` + worker crash = BrokenPipeError cascade | Restart loop with `done_paths` tracking |
| 12 | xterm has no clipboard copy for operator error capture | Use alacritty (mouse-select → clipboard) |
| 13 | bge-m3 encode-batch=512 OOM on 16 GB GPU | `--encode-batch 32` + OOM recovery loop |
| 14 | Two SentenceTransformer loads in one process → heap corruption | `mp.get_context('spawn')` subprocess isolation |

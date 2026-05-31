# Correctness — doc_classifier_gpu

Design rationale and correctness guide for the GPU embedding document classifier
pipeline. Written after 15 bugs across 18 commits — this document explains why the
final architecture is the way it is, and how to verify that a run actually did what
you think it did.

---

## §1 — What "correctness" means for this system

Before debugging a pipeline like this one, it is worth being precise about what we
mean by "correct." There are five independent dimensions:

| Dimension | Question it answers |
|-----------|---------------------|
| **Operational stability** | Does the pipeline run to completion without crashing or hanging? |
| **Data completeness** | Was every input file processed? Were any silently dropped? |
| **Output fidelity** | Is every classified file present in the CSV exactly once, with the right filename and prediction? |
| **ML accuracy** | Do the predictions match ground truth for the domain? |
| **Observability** | Can the operator see what the system is doing at any moment and detect failure? |

**The most important framing:** of the 15 bugs encountered building this system,
only 1 was an ML accuracy problem. The other 14 were failures in dimensions 1-3:
crashes, silent file drops, or deadlocks that produced no output at all. A classifier
that silently processes 40% of its input and drops the rest has zero useful accuracy
regardless of how good the model is.

A common mistake is to optimize for dimension 4 (ML accuracy) before dimensions 1-3
are solid. If you don't know whether all 26,577 files were classified, a 92% CV
accuracy score means nothing.

---

## §2 — The correctness stack

Correctness layers are ordered by dependency: lower layers gate upper layers.

```
┌─────────────────────────────────────────────────────────┐
│  Layer 5: ML accuracy                                   │
│           Model predicts the correct class              │
├─────────────────────────────────────────────────────────┤
│  Layer 4: Output fidelity                               │
│           CSV is complete, accurate, no duplicates      │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Data completeness                             │
│           Every input file reached the classifier       │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Operational stability                         │
│           Pipeline runs to completion without crashing  │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Observability                                 │
│           Operator can see what is happening            │
├─────────────────────────────────────────────────────────┤
│  Layer 0: Infrastructure                               │
│           Hardware, OS, IPC, and runtime work correctly │
└─────────────────────────────────────────────────────────┘
```

If layer 2 fails (pipeline crashes), layers 3-5 are irrelevant — there is no output.
If layer 3 fails (files silently dropped), your layer-5 accuracy metric is computed
on a biased sample. If layer 1 fails (you can't see what's happening), you may not
even know layers 2-3 have failed.

**Each section of this document addresses one or more layers.** Architecture decisions
in §4 explain which layer each decision protects.

---

## §3 — Failure taxonomy: 15 bugs, 6 categories

Every bug encountered building this system is catalogued here, categorized by type
and by which correctness layer it violated. The most important column is "Silent?" —
silent failures are the hardest because they produce no error output and appear to be
working normally.

| # | Bug | Category | Layer | Silent? |
|---|-----|----------|-------|---------|
| 1 | pdfplumber → Pillow SIGSEGV on corrupt PDFs | Crash / C-extension | 2 | No |
| 2 | ThreadPoolExecutor + GIL serializes pure-Python work | Silent serialization | 1 | **YES** |
| 3 | BrokenProcessPool poisons all pending futures | Crash isolation | 2 | Partial |
| 4 | `console.print(stderr=True)` is not a rich API param | API misuse | 1 | **YES** |
| 5 | macOS `._filename` resource forks crash EasyOCR | Data filtering | 3 | No |
| 6 | OOM at model load when GPU is busy | Resource exhaustion | 2 | No |
| 7 | Background subprocess hides all output | Observability | 1 | **YES** |
| 8 | Assumed one GPU at a time; second GPU idle | Resource awareness | 0 | No |
| 9 | Full OCR on bulk corpus = 91 hours | Scope correctness | 2 | No |
| 10 | MuPDF stderr warnings misread as failures | Observability noise | 1 | No |
| 11 | `imap_unordered` + `WorkerLostError` → BrokenPipe cascade | IPC / cascades | 2-3 | No |
| 12 | xterm has no clipboard copy for error capture | Observability | 1 | **YES** |
| 13 | encode-batch=512 OOM kills embed thread silently | Resource exhaustion | 2-3 | **YES** |
| 14 | Two SentenceTransformer loads → glibc heap corruption | IPC / runtime | 0 | No |
| 15 | spawn Queue + fork Pool → futex deadlock | IPC / concurrency | 2 | **YES** |

**6 of 15 bugs were silent.** They produced no error messages and appeared to be
running correctly. The only way to detect them was external observation: `mpstat`,
`nvidia-smi`, `ps -eo wchan`, or checking that output row count == input file count.

### Category breakdown

**Silent serialization (bug #2):** A process pool appears to have N workers but all
work serializes through one. Detection: `mpstat -P ALL 1` — only 1 core shows activity
despite N workers. Root cause: the GIL serializes threads for pure-Python code;
`ThreadPoolExecutor` is correct only for I/O-bound or C-extension work.

**Silent IPC deadlock (bug #15):** All processes are alive in `ps`, 0% CPU, 0% GPU,
no output, no error messages. Detection: `ps -eo pid,stat,wchan` — all show `futex_`
(waiting on a mutex). Root cause: Python's `multiprocessing.Queue` starts feeder
threads in the parent; forked child processes inherit locked feeder thread mutexes
with no thread to release them.

**Silent embed thread crash (bug #13):** CPU extraction is running (processes active,
disk I/O), but GPU utilization drops to 0%. Detection: `nvidia-smi` — GPU allocated
but 0% utilization; check terminal for "batch error:" messages. Root cause: unhandled
`torch.OutOfMemoryError` in a background thread kills the thread silently; the main
process continues extracting files that accumulate in the queue with no consumer.

**Silent data loss (bug #11):** Run appears to complete; CSV exists but has fewer rows
than expected. Root cause: when `imap_unordered` raises `WorkerLostError`, the `with
Pool` context manager calls `pool.terminate()` unconditionally, which closes all IPC
pipes; in-flight workers get `BrokenPipeError` and their results are lost.

---

## §4 — Why each architectural decision exists

For each key decision in the pipeline, this section explains: what it prevents, what
would happen without it, and which correctness layer it protects.

---

### CPU extraction layer

#### PyMuPDF (fitz), not pdfplumber

**Prevents:** SIGSEGV from Pillow C extensions in parallel workers.

**The problem without it:** `pdfplumber.page.extract_text()` calls
`pdfminer.image.PDFImageInterpreter` → `PIL.Image.open()` → `_imaging.so` (Pillow C
extension) for every PDF that has embedded images. A corrupt image byte sequence
triggers a segfault inside the C code. The worker process is killed by SIGSEGV. 98%
of the Liberated_manuals class (697 PDFs) have embedded images; with 14 workers, the
class is processing almost entirely sequentially due to constant segfaults.

**The fix:** `fitz.page.get_text()` handles embedded images internally via MuPDF's own
C layer. Pillow is never invoked. Smoke test confirmed: a 312-page, 68 MB PDF with
hundreds of embedded images extracted 720,000 chars in 0.31 seconds with no crash.

**Layer protected:** 2 (operational stability), 3 (data completeness — segfaults
cause some files to be silently skipped).

---

#### ProcessPool, not ThreadPool

**Prevents:** GIL serialization of pure-Python work.

**The problem without it:** `pdfminer`, `pdfplumber`, and `pypdf` have **zero C
extension files** (confirmed by `find ~/.local/lib -name "*.so" | grep pdfminer`).
They are pure Python. Python's GIL (Global Interpreter Lock) allows only one thread
to execute Python bytecode at a time. `ThreadPoolExecutor` with 14 workers gives
14× process-management overhead with 1× actual extraction throughput. `mpstat`
confirmed: only 1 core busy during "14-worker parallel" extraction.

**The fix:** `multiprocessing.Pool` spawns separate Python interpreters. Each
interpreter has its own GIL. True parallelism: 12 cores extract simultaneously.

**Rule:** Threads parallelize I/O-bound work and C-extension work. For pure-Python
CPU-bound work, use processes.

**Layer protected:** 1 (observability — the system appeared to be working), 2
(operational stability — a 14-hour extraction job that runs in 1 hour instead of 14).

---

#### `Pool(maxtasksperchild=1)`, not default

**Prevents:** one worker crash poisoning all pending futures.

**The problem without it:** With `ProcessPoolExecutor` (or `Pool` with no
`maxtasksperchild`), a worker handles many tasks before being replaced. When a worker
crashes on task 7, the pool raises `BrokenProcessPool` for ALL remaining tasks
submitted to that pool object — not just task 7. Everything in-flight and pending is
marked as failed. With 697 files in Liberated_manuals and 14 workers, one crash at
file 7 causes files 8-697 to fall back to sequential processing.

**The fix:** `maxtasksperchild=1` causes each worker to exit cleanly after handling
one file. A crash affects exactly one file. The pool spawns a fresh worker for the
next. There is no "pending futures" accumulation. Fork is fast on Linux (<1ms), so
the overhead is negligible.

**Layer protected:** 2 (operational stability — crash isolation), 3 (data
completeness — without isolation, one crash could drop hundreds of files).

---

#### imap_unordered restart loop with `done_paths`

**Prevents:** `WorkerLostError` from silently abandoning unprocessed files.

**The problem without it:** `pool.imap_unordered()` raises `WorkerLostError` when a
worker dies unexpectedly. The `with Pool(...) as pool:` context manager calls
`pool.terminate()` unconditionally when exiting — whether via exception OR normal
exit. Terminating the pool closes all IPC pipes. Every in-flight worker trying to
send its result gets `BrokenPipeError`. The cascade generates hundreds of tracebacks.
All files that hadn't completed yet are simply abandoned — no error, no record in CSV,
no indicator of loss.

**The fix:**
```python
done_paths = set()
while files_todo and not _shutdown.is_set():
    with Pool(...) as pool:
        try:
            for path_str, text in pool.imap_unordered(_extract_cpu, files_todo):
                done_paths.add(path_str)
                ...
        except Exception as exc:
            console.print(f"⚠ Worker crash: {exc}")
    files_todo = [p for p in files_todo if p not in done_paths]
    # restart with only the remaining files
```

The `done_paths` set is the correctness invariant: it tracks every file that returned
a result. After each pool exit (crash or normal), we compute the difference. Files
not in `done_paths` are retried. Files that crash `MAX_FAILS=2` times are blacklisted
to `_review/`.

**Layer protected:** 3 (data completeness — no file is silently abandoned).

---

### GPU embedding layer

#### `mp.get_context('spawn')` for embed subprocesses

**Prevents:** glibc heap corruption from dual SentenceTransformer loads.

**The problem without it:** Calling `SentenceTransformer(model, device="cuda:0")`
then `SentenceTransformer(model, device="cuda:1")` in the same process corrupts
the glibc heap (crash: `free(): corrupted size vs. prev_size`, SIGABRT). The root
cause is the HuggingFace `tokenizers` library's Rust FFI layer, which maintains
global state that cannot be initialized twice in one process. Additionally, PyTorch
explicitly documents that forking after CUDA initialization is unsafe.

**The fix:** `mp.get_context('spawn')` starts each embed subprocess as a fresh Python
interpreter. Each subprocess has its own:
- glibc heap (no sharing with parent or sibling subprocess)
- HuggingFace tokenizer Rust FFI state (initialized once, cleanly)
- CUDA context (initialized fresh, no fork-after-CUDA issue)

**Layer protected:** 0 (infrastructure — SIGABRT is the lowest-level failure).

---

#### `mp.Manager().Queue()`, not `ctx.Queue()`

**Prevents:** spawn+fork deadlock — the most insidious failure in this system.

**The problem without it:** Python's `multiprocessing.Queue` starts a background
feeder thread in the process that creates it. This thread handles the actual pipe
writes asynchronously. When `Pool()` then forks 12 extraction workers, those workers
inherit the parent's memory — **including mutexes held by the feeder thread at the
moment of fork**. Fork copies memory but not threads. The feeder threads do not exist
in the forked children. Any lock the feeder thread held at fork time is permanently
locked in every child.

The child processes call `imap_unordered` which uses Pool's internal result queue.
At some point this involves acquiring the same family of synchronization primitives
that are already locked. Every worker deadlocks on `futex_`. No error message. No
output. Processes appear alive in `ps`. CPU 0%. GPU 0%. Nine hours and thirty-seven
minutes of nothing.

**Confirmed diagnosis:**
```bash
ps -eo pid,stat,wchan,cmd | grep sort_docs
# Output: all processes in S state, wchan=futex_
# nvidia-smi: GPU 0 allocated (13.8 GB), 0% utilization
# nvidia-smi: GPU 1 allocated (9.5 GB), 0% utilization
# No CSV output after 9h37m
```

**The fix:** `mp.Manager()` spawns a separate server process. `manager.Queue()` returns
a proxy object. `put()` and `get()` communicate with the server via socket. No feeder
threads in the main process. No shared mutexes. Pool workers forked from the main
process inherit only a socket file descriptor — which is safe to inherit.

```python
ctx     = mp.get_context('spawn')
manager = mp.Manager()                       # server process, no feeder threads
in_qs   = [manager.Queue(), manager.Queue()] # proxy objects — fork-safe
out_qs  = [manager.Queue(), manager.Queue()]
```

**Why not use spawn for the extraction Pool too?** Spawn takes ~30s to start 12
workers (fresh Python interpreter per worker). Fork takes <1ms. We restart the
extraction Pool on crashes; with spawn, each restart adds 30s overhead. Manager
queues solve the fork-safety problem without spawn overhead for the extraction layer.

**Layer protected:** 2 (operational stability — deadlock produces zero output), 3
(data completeness — no files are classified at all during a deadlock).

---

#### OOM recovery in `embed_batch()`

**Prevents:** silent embed thread crash from large encode batches.

**The problem without it:** bge-m3 uses ~13 GB on the RTX 5080 (16 GB total), leaving
~1.2-1.5 GB for activations. With `encode_batch=512`, a forward pass on 512 text
chunks simultaneously requires 2.94 GB → `torch.OutOfMemoryError`. This exception is
raised inside a background thread (`_embedding_worker`). An unhandled exception in a
daemon thread terminates that thread silently — the main process continues extracting
files and accumulating them in the embed queue with no consumer. The queue grows
unboundedly. GPU sits at 0% utilization. No error message is shown.

**The fix:**
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
        print(f"⚠ GPU OOM — retrying encode_batch={encode_batch}", file=sys.stderr)
```

The embed subprocess catches OOM, halves the batch size, clears the CUDA cache, and
retries. It degrades gracefully rather than dying. Default `--encode-batch 32` is
already safe for bge-m3 on 16 GB; the recovery loop is belt-and-suspenders.

**Layer protected:** 2-3 (if the embed thread crashes, subsequent files are silently
unclassified).

---

### Classification layer

#### LogisticRegression with calibrated probabilities

**Why not a neural classifier?** With 15-100 documents per class, a neural classifier
overfits. LogisticRegression with `class_weight="balanced"` generalizes well at this
scale. The `predict_proba()` output is calibrated — a score of 0.40 means "roughly
40% confident," which makes the `--threshold` flag semantically meaningful.

**The weight norm as a quality signal:** `np.linalg.norm(clf.coef_[i])` for each class
gives a direct measure of how distinctively embedded that class is. High norm = strong
cluster in embedding space = reliable predictions. Low norm = class overlaps with
others = consider merging with a similar class or adding more training examples.

**Why not SVM?** SVM produces margin scores, not probabilities. A threshold on margin
scores has no interpretable meaning. LogReg produces probabilities; a threshold of
0.40 is operationally meaningful to a human reviewer.

#### Mean pooling over chunk embeddings

**Why not truncate to the first N tokens?** A 40-page field manual has important
content on page 1 (title, introduction) AND page 38 (appendix, references). Truncation
to 512 tokens drops approximately 90% of the document.

**Why not use the maximum-scoring chunk?** Max pooling is biased toward the most
"topically pure" chunk of the document. A manual on field sanitation with a section
on weapons handling would score as either "sanitation" OR "weapons" — but the correct
class should reflect the whole document. Mean pooling dilutes outlier chunks and
produces a balanced representation.

**Overlapping chunks (200-char overlap):** A sentence split exactly at a 4000-character
chunk boundary loses context in both halves. The 200-character overlap ensures that
every sentence appears complete in at least one chunk.

#### bge-m3 over TF-IDF

**When TF-IDF is better:** When document classes differ primarily by vocabulary —
legal vs. financial vs. personal. TF-IDF is faster, simpler, requires no GPU, and
matches or beats neural embeddings when the vocabulary signal is clean.

**When bge-m3 is better:** When classes share vocabulary but differ in meaning,
context, or use — which is the case for military documents. All military documents
use "weapons," "command," "logistics," and "intelligence." A TF-IDF classifier on
this corpus learns that these words appear everywhere and assigns them low weight —
losing exactly the signal that distinguishes classes. bge-m3 understands that
"tactical intelligence assessment" and "strategic intelligence overview" are different
concepts even though they share "intelligence." The embedding space reflects the
difference.

---

## §5 — The completeness invariant

**A sort run is only correct if output rows = input files.**

This is the single most important verification to perform after any run:

```bash
#!/bin/bash
# completeness_check.sh
SOURCE="$1"      # e.g. /path/to/usafa_af_mil
CSV="$2"         # e.g. ./usafa_sorted/sort_report.csv

INPUT=$(find "$SOURCE" -type f \
  ! -name "._*" \
  \( -name "*.pdf" -o -name "*.txt" -o -name "*.docx" \
     -o -name "*.jpg" -o -name "*.png" \) \
  | wc -l)

if [ ! -f "$CSV" ]; then
    echo "FAIL: CSV not found — run did not complete"
    exit 1
fi

CSV_ROWS=$(wc -l < "$CSV")
CLASSIFIED=$((CSV_ROWS - 1))   # minus header row

echo "Input files:   $INPUT"
echo "CSV rows:      $CSV_ROWS  (incl. header)"
echo "Classified:    $CLASSIFIED"
echo "Delta:         $((INPUT - CLASSIFIED))"

if [ "$INPUT" -eq "$CLASSIFIED" ]; then
    echo "PASS: all files accounted for"
else
    echo "FAIL: $(( INPUT - CLASSIFIED )) files missing from output"
    echo "      Check for: 'Gave up on N files', 'Partial run', or embed crashes"
fi
```

### Interpreting the results

**Perfect run:** `delta = 0`. Every file either has a predicted class (confidence ≥
threshold) or is in `_review/` (no text or confidence < threshold). Both are correct
outcomes — `_review/` is not a failure, it is the system correctly saying "I can't
classify this reliably."

**Partial run:** `delta > 0`. Some files were not classified. Causes ranked by
likelihood:
1. Ctrl-C or SIGTERM during the run (check for "Partial run" banner in terminal)
2. Persistent crash loop: `maxtasksperchild` crashed repeatedly on same file(s)
   (check for "Gave up on N files" messages)
3. Embed subprocess crash: background thread died mid-run (check for "batch error:"
   in embed subprocess stderr)
4. Network drive disconnected mid-run (check mount point still accessible)

**To recover:** Rerun the same command. The `done_paths` tracking is reset between
runs, but `sort_docs.py` in `--mode report` will simply overwrite the CSV. The restart
loop doesn't persist state between process invocations — if you need incremental
resumption for very large collections, track `done_paths` externally.

---

## §6 — Silent failure detection guide

During a run, these commands tell you whether each layer is actually working.

### Is CPU extraction actually parallel?

```bash
mpstat -P ALL 2 3
```

**Expected:** Multiple cores showing 30-80% `%usr` during extraction phase.
**Bad sign:** Only 1 core at high CPU while others idle.
**Diagnosis:** If the executor uses `ThreadPoolExecutor` and the work is pure Python,
the GIL serializes all threads. Check that the extraction pool uses `Pool(processes=N)`.

### Is the GPU actually computing?

```bash
watch -n 2 'nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used --format=csv,noheader'
```

**Expected:** During embedding phase, GPU(s) show 50-100% utilization.
**Bad sign:** GPU allocated (memory.used shows GB) but utilization 0%.
**Diagnoses:**
- All `futex_` in `ps wchan` → spawn+fork deadlock (see §4)
- GPU was active then dropped to 0% → embed thread crashed (OOM or other exception)
- GPU never activated at startup → embed subprocess failed to start (check terminal
  for "✓ [cuda:N] embed subprocess ready" message)

### Are processes deadlocked?

```bash
ps -eo pid,stat,wchan,cmd | grep sort_docs | head -30
```

**Expected states:**
- `S` on `futex_` = waiting for Python lock (normal for idle workers)
- `R` = actively running
- `S` on `pipe_r` = waiting to read from pipe (normal for consumer threads)

**Deadlock signature:**
```
ALL processes in S state on futex_
No progress in output (wc -l sort_report.csv)
0% CPU, 0% GPU
Processes alive for hours
```

This is the spawn+fork deadlock (bug #15). Kill the run and ensure `mp.Manager().Queue()`
is used for spawn subprocess IPC (not `ctx.Queue()`).

### Did the embed subprocess start correctly?

In the alacritty window, within 90 seconds of launch, you should see:
```
  ✓ [cuda:0] embed subprocess ready
  ✓ [cuda:1] embed subprocess ready
```

If these messages don't appear within 90s:
- Subprocess failed to import dependencies (check for ImportError)
- Model download failed (no internet, or HuggingFace quota exceeded)
- VRAM probe failed (GPU not visible from subprocess)

If they appeared but the GPU later dropped to 0%: the embed subprocess encountered
an unrecoverable error after starting. The OOM recovery loop handles `OutOfMemoryError`;
for other errors check the subprocess stderr output.

### Is the run making forward progress?

```bash
# Poll every 30s during a run
while true; do
    if [ -f usafa_sorted/sort_report.csv ]; then
        echo "$(date) — $(wc -l < usafa_sorted/sort_report.csv) rows"
    else
        echo "$(date) — CSV not created yet"
    fi
    sleep 30
done
```

**Note:** The CSV is written only at run completion. During the run, the progress bar
in alacritty shows `task_extract` and `task_embed` advancing in real-time. If those
bars are not moving for >5 minutes, the run is stalled.

---

## §7 — Why this architecture and not simpler alternatives

### "Why not just use a single process?"

C extension libraries (Pillow, libfitz, CUDA) can `SIGSEGV` on corrupt input data. A
segfault in a single-process pipeline kills the entire job immediately. Separate worker
processes isolate crashes: the worker dies, the pool detects the failure, the file is
flagged, and processing continues. Without process isolation, one corrupt PDF in a
26,000-file corpus halts the entire run.

### "Why not use threads for CPU extraction?"

pdfminer, pdfplumber, and pypdf are pure Python with zero C extension files. The GIL
serializes all threads that execute pure Python bytecode. `ThreadPoolExecutor(14)` gives
exactly 1× actual throughput with 14× overhead. `ProcessPool(14)` gives approximately
14× throughput because each interpreter has its own GIL. Always profile with `mpstat`
before assuming threads are helping.

### "Why not use spawn for the extraction Pool too?"

`spawn` starts a fresh Python interpreter per worker. Startup time is ~2-3 seconds per
worker on this machine. With 12 workers: ~25s startup per pool creation. With `fork`,
startup is <1ms total for all 12. Since the extraction pool restarts on crashes (the
restart loop creates a new `Pool()` object), using spawn adds 25s penalty per crash
recovery. Fork is safe for the extraction pool as long as we use `Manager().Queue()`
for the spawn subprocess IPC — which eliminates the feeder-thread lock inheritance issue.

### "Why not use the same Queue type everywhere?"

There are three distinct Queue types in this pipeline:

| Queue | Used for | Fork-safe? | Thread-safe? |
|-------|----------|------------|--------------|
| `threading.Queue` | Single-GPU embed thread | N/A (threads only) | Yes |
| `mp.Manager().Queue()` | Dual-GPU spawn subprocess IPC | **YES** | Yes |
| `mp.get_context('spawn').Queue()` | (what we used before fixing bug #15) | **NO** | Yes |

`ctx.Queue()` appears to be the right choice for spawn subprocesses. But it starts
feeder threads in the creating process. When the creating process later forks extraction
workers, those workers inherit locked feeder thread mutexes. Deadlock. `Manager().Queue()`
avoids this by routing all IPC through a server process via socket — no shared state,
no feeder threads in the main process.

### "Why LogisticRegression over fine-tuning the encoder?"

Fine-tuning bge-m3 end-to-end requires contrastive pairs (anchor, positive, negative)
or classification labels with at minimum hundreds of examples per class. This corpus
has 15-100 documents per class. Fine-tuning on this data would overfit severely. The
pretrained bge-m3 already encodes semantic similarity well; a LogisticRegression head
learns the decision boundary in the embedding space with minimal data.

If the corpus grows to 500+ examples per class, fine-tuning becomes viable. The path
is: replace `get_encoder()` with a `transformers` `AutoModel`, add a training loop,
keep all extraction and chunking logic unchanged.

### "Why skip OCR by default for bulk classification?"

EasyOCR at 2× render scale processes approximately 1 page per 3 seconds on the RTX
3080. A scanned manual with 200 pages takes 10 minutes. 1,664 image-only PDFs in the
USAFA corpus × 10 minutes = approximately 277 hours. The goal of bulk sorting is
classification, not full-text indexing. `--skip-ocr` sends image-only files to
`_review/` and completes the rest in ~14 minutes instead of 277 hours.

For the `_review/` subset, use `--max-ocr-pages 3` to sample the first 3 pages of
each file — enough to read the title, table of contents, and opening section for
classification purposes, in approximately 30-60 seconds per file.

### "Why two GPUs instead of one?"

For 26,577 files with bge-m3, single-GPU sorting takes approximately 14 minutes. Dual-GPU
reduces this to approximately 7 minutes. For larger corpora — millions of files,
which is the scale of the PDF archive described in the project context — the ~2×
throughput scales linearly: a 10-hour job becomes a 5-hour job.

The dual-GPU implementation has non-trivial correctness requirements (spawn isolation,
Manager queues) precisely because GPU model loading has strong constraints. The
single-GPU path remains simpler and is the default (`--single-gpu`). The dual-GPU
path is opt-in (`--no-single-gpu`) after the architecture was validated.

---

## §8 — Building a similar system: decision guide

### Architecture decisions

When extracting text from PDF/image/document files in parallel:

```
Does the extraction library use Pillow or other C extensions?
  YES → Use PyMuPDF (fitz) for PDFs; it handles images internally
  NO  → Can use any library, but audit the .so files first

Are you using Python threads or processes for parallelism?
  Using threads with pure-Python libraries → switch to ProcessPool
  Using threads with C-extension libraries → threads may help (release GIL)
  Using processes → use maxtasksperchild=1 for crash isolation

Will you be mixing spawn subprocesses with fork pools?
  YES → Use mp.Manager().Queue() for cross-context IPC
        Do NOT use ctx.Queue() (creates feeder threads → fork deadlock)
  NO  → Use threading.Queue (same-process) or ctx.Queue (spawn only)
```

When loading GPU models in multiple processes:

```
Are you loading two instances of the same model?
  In the same process → WILL cause heap corruption (Rust FFI global state)
  Use spawn subprocesses → each gets its own heap; safe
  
Are you forking after CUDA initialization?
  YES → CUDA explicitly prohibits this (PyTorch docs)
  Use spawn for any process that will use CUDA
```

When building a streaming extraction+embedding pipeline:

```
CPU extraction is the bottleneck → increase --workers
GPU embedding is the bottleneck → use --no-single-gpu (if 2+ GPUs available)
Both are bottlenecks → increase both, check completeness invariant after

Always add OOM recovery to encoder.encode():
  - Catch (RuntimeError, torch.cuda.OutOfMemoryError)
  - Halve batch size on OOM, clear CUDA cache, retry
  - Minimum batch size of 4
```

### Completeness verification checklist

Before declaring a sort run correct:

```bash
# 1. Row count check (most important)
wc -l sort_report.csv
# Should equal: input file count + 1 (header)

# 2. GPU utilization during run
# Both GPUs should have shown >50% utilization
# (Check terminal output or session logs)

# 3. No silent failure indicators
grep -c "Gave up on" <terminal_log>   # should be 0
grep -c "Partial run" <terminal_log>  # should be 0
grep -c "batch error" <terminal_log>  # should be 0

# 4. Class distribution sanity check
awk -F',' 'NR>1{print $2}' sort_report.csv | sort | uniq -c | sort -rn | head -15
# No single class should be >60% of total (unless corpus is genuinely skewed)

# 5. _review percentage
grep -c ",_review," sort_report.csv
# For --skip-ocr runs: expected 5-20% (image-only files + low-confidence files)
# >50% suggests: threshold too high, model mismatch, or most files are image-only
```

### Operational checklist

```
[ ] Use alacritty (not xterm) — mouse-select → clipboard for error copying
[ ] Launch in visible terminal (not background) — training/sorting output is observability
[ ] Run completeness check after every sort
[ ] Monitor both GPUs during dual-GPU run (watch nvidia-smi)
[ ] Check "✓ [cuda:N] embed subprocess ready" appears within 90s of launch
[ ] For first run on a new corpus: start with --mode report before --mode copy
[ ] Use --skip-ocr for first pass; OCR selectively on _review/ subset
[ ] Keep --encode-batch ≤ 64 for bge-m3 on 16 GB GPU
```

---

## §9 — What this system does NOT guarantee

Understanding the limits of correctness is as important as understanding the design.

**ML accuracy is not validated externally.** The 5-fold CV accuracy reflects how well
the model generalizes within the training distribution. Documents that are genuinely
ambiguous between two classes will be placed in whichever class has the higher
embedding similarity — which may not match a human expert's judgment.

**`_review/` is not a correctness failure.** Files in `_review/` either had no
extractable text (correct: we have no signal to classify them) or scored below
the confidence threshold (correct: we are uncertain and say so). A high `_review/`
percentage indicates a corpus mismatch (most files differ from the training domain),
not a pipeline bug.

**The training corpus defines the label space.** The classifier knows exactly 64
classes. A document from outside that label space will be assigned the "nearest"
class in embedding space, not flagged as out-of-distribution. If you sort a corpus
that contains document types not in the training data, confidence scores will be
lower and `_review/` percentage will be higher — but no explicit warning is issued.

**Extraction quality affects embedding quality.** A scanned PDF where OCR produces
garbled text will produce a distorted embedding. The classifier will assign a class
based on whatever text was extracted, which may not reflect the document's actual
content. For image-only PDFs without OCR, the file goes to `_review/` — which is
the correct behavior (no signal → no prediction).

---

## Summary: the correctness hierarchy in one table

| Correctness property | How it's achieved | How to verify |
|---------------------|-------------------|---------------|
| No SIGSEGV crashes | PyMuPDF (no Pillow) | Run trains/sorts without BrokenProcessPool |
| True CPU parallelism | ProcessPool (not ThreadPool) | mpstat shows N cores busy |
| Crash isolation | maxtasksperchild=1 | Individual file failures don't abort the run |
| No silent file loss | imap_unordered restart loop | CSV row count == input file count |
| No spawn+fork deadlock | Manager().Queue() | Progress bar advances within 60s |
| No silent embed crash | OOM recovery loop | GPU stays active during embedding phase |
| No heap corruption | spawn subprocess isolation | Both GPUs load without SIGABRT |
| Operator visibility | alacritty, rich TUI, GPU stats | Errors are copyable; progress is visible |
| Data completeness | completeness invariant check | Run completeness_check.sh after every sort |
| ML calibration | LogisticRegression, balanced weights | CV accuracy + per-class weight norms |

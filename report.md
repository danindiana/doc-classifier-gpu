# Concurrency and Parallelism Analysis: `doc-classifier-gpu`

This report provides a comprehensive analysis of the complex concurrency negotiations, parallel processing strategies, and fault tolerance mechanisms employed in the `doc_classifier_gpu.py` and `sort_docs.py` scripts. 

## 1. The Architecture of Parallelism
The project implements a multi-stage, hybrid CPU/GPU parallel processing pipeline designed to maximize resource utilization while mitigating the fragility of Python's multiprocessing and third-party C-extensions. 

The core challenge addressed is the disparity between **CPU-bound operations** (text extraction via parsing) and **GPU-bound operations** (transformer-based embedding and OCR). If handled sequentially, the GPU would starve while waiting for CPU extraction, and vice-versa.

### 1.1. CPU Parallelism (Text Extraction)
Text extraction from complex documents (especially PDFs) is highly CPU-intensive and prone to crashes (e.g., segfaults in underlying C libraries when encountering corrupt embedded images). 

- **Implementation**: The pipeline uses `multiprocessing.pool.Pool` to spread extraction across multiple CPU cores.
- **Task Isolation (`maxtasksperchild`)**: Both scripts utilize `maxtasksperchild` (defaulting to 20 or 50). This mitigates memory leaks in external libraries by recycling worker processes after a set number of tasks. Setting it to 1 provides maximum isolation at the cost of fork overhead.

### 1.2. Decoupled Pipeline (Streaming Architecture)
In `sort_docs.py`, the pipeline is truly decoupled:
- The main thread continuously feeds and drains a `pool.imap_unordered` iterator. This ensures CPU workers never stall.
- Batches of extracted text are pushed to thread-safe/process-safe queues.
- A background worker (thread or subprocess) consumes these queues to perform GPU embedding simultaneously.

## 2. Esoteric Concurrency Negotiations & Fault Tolerance

The scripts exhibit deep awareness of Python's GIL, multiprocessing caveats, and native library interactions.

### 2.1. The `BrokenPipeError` / `WorkerLostError` Cascade
**The Problem**: If a worker process in a standard `Pool` crashes (e.g., a SIGSEGV triggered by `fitz` reading a corrupt PDF), the pool is poisoned. `imap_unordered` throws a `WorkerLostError`, which, if unhandled, causes the `with Pool` context to exit, calling `pool.terminate()`. This abruptly closes IPC pipes, throwing `BrokenPipeError` across all remaining active workers.
**The Solution**: `sort_docs.py` implements a sophisticated **Restart Loop**.
1. It maintains a set of `done_paths`.
2. A generator loops over the remaining `files_todo`.
3. If an exception breaks the pool, the main thread catches it, registers the crashed files, increments a `fail_counts` tracker, and **spawns a brand new `Pool`** to resume processing the remaining `files_todo`.
4. Files that crash repeatedly (hitting `MAX_FAILS`) are gracefully abandoned to a `no_text_files` list, preventing infinite crash loops.

### 2.2. Dual-GPU Subprocess Isolation (Heap Corruption Prevention)
**The Problem**: Attempting to load multiple `SentenceTransformer` models (one per GPU) in different threads of the *same* Python process leads to glibc heap corruption.
**The Solution**: The `--no-single-gpu` flag employs process-level isolation:
1. `mp.get_context('spawn')` is explicitly used. Forking an existing multi-threaded process is dangerous; `spawn` guarantees a fresh Python interpreter and memory space for each GPU worker.
2. Each subprocess loads its own `SentenceTransformer` and `LogisticRegression` model, bound to `cuda:0` or `cuda:1`.

### 2.3. Queue Deadlock Avoidance (`Manager.Queue` vs `ctx.Queue`)
In Dual-GPU mode, the script avoids a severe multiprocessing footgun:
- Using `ctx.Queue()` starts background feeder threads in the main process. If the main process subsequently forks (e.g., when instantiating the CPU `Pool()`), the child processes might inherit locked mutexes from those feeder threads, leading to silent, permanent deadlocks (workers stuck waiting on `futex_`).
- **The Fix**: The script deliberately uses `mp.Manager().Queue()`. Manager queues utilize a separate server process and communicate via sockets. This eliminates shared mutex locks and guarantees fork-safety for the CPU workers spawned later.

### 2.4. GPU OOM (Out-of-Memory) Recovery
**The Problem**: The BAAI/bge-m3 model occupies ~13.4 GB of the RTX 5080's 16 GB VRAM. Large batch sizes of highly dense text chunks can easily exceed the remaining memory.
**The Solution**: A graceful degradation loop in `embed_batch`. If `torch.cuda.OutOfMemoryError` is caught during a forward pass:
1. It calls `torch.cuda.empty_cache()`.
2. It halves the `encode_batch` size dynamically.
3. It retries the batch, continuing to halve the batch size (down to a hardcoded minimum of 4) until the forward pass succeeds. 

### 2.5. Graceful Preemption (Signal Handling)
A global `_shutdown` event handles `SIGINT` and `SIGTERM`. Instead of immediately terminating, it halts the generation of new tasks, signals the pool to terminate cleanly, drains the pending queues, and persists a partial CSV report.

## 3. Summary
The logic transcends basic parallelization. It acknowledges that in a high-throughput, multi-modal pipeline, hardware limitations (VRAM) and software fragility (C-extension segfaults) are certainties, not anomalies. By leveraging `spawn` contexts, socket-based manager queues, dynamic batch scaling, and active pool-restart loops, the system achieves maximum throughput while remaining virtually crash-proof.

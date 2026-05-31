# Implementation Guide: Advanced Concurrency & Fault Tolerance in Python

This guide details exactly how to implement the advanced concurrency, parallelization, and fault-tolerance patterns analyzed in the original report. These patterns are essential for robust machine learning pipelines that hybridize CPU-bound tasks (like parsing) and GPU-bound tasks (like embedding and inference).

## 1. CPU Extraction with `maxtasksperchild`
When using third-party C-extensions (like PyMuPDF or Pillow), memory leaks or segmentation faults can occur over time. To prevent a long-running worker from gradually leaking memory, recycle workers after a certain number of tasks.

**Implementation Details:**
Use `multiprocessing.pool.Pool` (or `ProcessPoolExecutor` via `mp.get_context()`) with the `maxtasksperchild` argument.

```python
import multiprocessing as mp
from multiprocessing.pool import Pool

def extract_text(filepath):
    # Fragile C-extension operations go here
    pass

# maxtasksperchild=50 forces the Pool to kill and respawn a worker 
# process after it has processed 50 files. This isolates memory leaks.
with Pool(processes=8, maxtasksperchild=50) as pool:
    for result in pool.imap_unordered(extract_text, files, chunksize=1):
        process(result)
```

## 2. Resilience to Pool Poisoning (The Restart Loop)
If a worker crashes abruptly (e.g., a SIGSEGV), the active `Pool` is poisoned and throws exceptions like `WorkerLostError` or `BrokenPipeError`. To make your pipeline bulletproof, catch the exception and spawn a new pool for the remaining files.

**Implementation Details:**
Maintain a set of completed tasks, a tracker for task failures, and wrap the `Pool` inside a `while` loop.

```python
files_todo = list(all_files)
done_paths = set()
fail_counts = {f: 0 for f in files_todo}
MAX_FAILS = 2

while files_todo:
    try:
        with Pool(processes=8, maxtasksperchild=50) as pool:
            for filepath, text in pool.imap_unordered(extract_text, files_todo, chunksize=1):
                done_paths.add(filepath)
                # Process the successfully extracted text...
    except Exception as e:
        print(f"Worker crashed: {e}. Restarting pool...")
        
    # Filter out completed files
    files_todo = [p for p in files_todo if p not in done_paths]
    
    if files_todo:
        # Increment fail counts for remaining files (one of them caused the crash)
        for p in files_todo:
            fail_counts[p] += 1
            
        # Give up on files that crash repeatedly to prevent infinite loops
        bad_files = [p for p in files_todo if fail_counts[p] >= MAX_FAILS]
        for bad in bad_files:
            done_paths.add(bad)
            
        files_todo = [p for p in files_todo if fail_counts[p] < MAX_FAILS]
```

## 3. Subprocess Isolation for Multi-GPU Systems
Loading large models (like `SentenceTransformer`) multiple times within threads of the same Python process can lead to glibc heap corruption.

**Implementation Details:**
Use `mp.get_context('spawn')` to guarantee a completely fresh Python interpreter for each GPU worker, rather than relying on `fork` (the Linux default).

```python
import multiprocessing as mp

def gpu_worker(gpu_id, in_q, out_q):
    # This runs in a completely isolated process space
    model = load_model(device=f"cuda:{gpu_id}")
    while True:
        batch = in_q.get()
        if batch is None:
            break
        results = model.predict(batch)
        out_q.put(results)

# Force 'spawn' start method
ctx = mp.get_context('spawn')

# Create isolated processes
procs = []
for i in range(num_gpus):
    p = ctx.Process(target=gpu_worker, args=(i, in_qs[i], out_qs[i]))
    p.start()
    procs.append(p)
```

## 4. Fork-Safe Queues via `Manager()`
When combining `spawn`-based GPU workers with a `fork`-based CPU `Pool`, do NOT use standard `multiprocessing.Queue()`. Standard queues use background feeder threads that hold mutex locks. If the main process forks while a lock is held, the child process inherits the locked mutex but not the thread that unlocks it, causing a permanent deadlock.

**Implementation Details:**
Use `mp.Manager().Queue()`. The Manager creates a separate server process that handles queue operations via Unix sockets, completely bypassing the shared memory and mutex deadlocks.

```python
import multiprocessing as mp

# CORRECT: Socket-based queues, completely fork-safe
manager = mp.Manager()
in_q = manager.Queue()
out_q = manager.Queue()

# WRONG: Will deadlock if you later call `Pool()` in the main process
# ctx = mp.get_context('spawn')
# in_q = ctx.Queue() 
```

## 5. Graceful GPU Out-Of-Memory (OOM) Recovery
If a batch size is too large for the available VRAM, standard ML scripts crash. Instead, catch the OOM error, clear the cache, dynamically halve the batch size, and retry.

**Implementation Details:**
Use a `while` loop that intercepts `torch.cuda.OutOfMemoryError` or `RuntimeError` containing "out of memory".

```python
import torch

def embed_with_oom_recovery(model, data, encode_batch=512):
    while True:
        try:
            # Attempt forward pass
            vectors = model.encode(data, batch_size=encode_batch)
            return vectors
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # Check if the error is actually an OOM
            if "out of memory" not in str(e).lower() or encode_batch <= 4:
                raise e # Real error or batch size is critically low
                
            # Recovery steps
            torch.cuda.empty_cache()
            encode_batch = max(4, encode_batch // 2)
            print(f"GPU OOM intercepted. Retrying with batch size {encode_batch}...")
```

## 6. Graceful Process Preemption (Ctrl+C)
Instead of hard-killing the script and losing processed data, intercept `SIGINT` and `SIGTERM` to safely drain queues and save partial results.

**Implementation Details:**
Use a global `threading.Event` as a shutdown flag.

```python
import signal
import threading

shutdown_flag = threading.Event()

def handle_signal(signum, frame):
    print("Graceful shutdown initiated. Finishing current batch...")
    shutdown_flag.set()

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# In your main extraction loop:
while files_todo and not shutdown_flag.is_set():
    # Process files...
    pass
    
# After the loop, safely write whatever was processed to disk
save_partial_results()
```

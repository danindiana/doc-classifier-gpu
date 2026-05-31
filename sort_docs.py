#!/usr/bin/env python3
"""Bulk-classify and sort a document collection using a trained militia.joblib model.

Usage:
    python sort_docs.py SOURCE_DIR \\
        --model militia.joblib \\
        --output ./sorted_output \\
        --mode copy            # copy | symlink | report
        --threshold 0.60       # confidence < threshold → _review/
        --workers 24           # parallel CPU extraction processes
        --maxtasks 20          # pool maxtasksperchild (1=max isolation)
        --batch 512            # documents per streaming window
        --encode-batch 32      # encoder.encode internal GPU batch size (32 safe for bge-m3)
        --skip-ocr             # skip GPU OCR (fast; image files → _review/)
        --no-single-gpu        # use both GPUs via subprocess isolation (~2× throughput)

Output:
    sorted_output/
    ├── strategy/          ← predicted class ≥ threshold
    ├── medical/
    ├── ...
    ├── _review/           ← max confidence < threshold or no text
    └── sort_report.csv    ← full results table
"""

import argparse
import csv
import multiprocessing as mp
import os
import queue
import shutil
import signal
import sys
import threading
import time
from collections import Counter
from multiprocessing.pool import Pool
from pathlib import Path

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# ── Graceful shutdown ──────────────────────────────────────────────────────────
_shutdown = threading.Event()

def _handle_signal(signum, frame):
    if not _shutdown.is_set():
        try:
            from doc_classifier_gpu import console
            console.print(
                "\n  [yellow]⚠ Ctrl-C — finishing current window then saving "
                "partial results (Ctrl-C again to force quit)[/]")
        except Exception:
            print("\n  ⚠ Shutdown requested — saving partial results...")
    _shutdown.set()

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

from doc_classifier_gpu import (
    _extract_cpu,
    _gpu_stat,
    _make_progress,
    chunk_text,
    console,
    extract_text,
    get_encoder,
    get_ocr_reader,
)

from rich.panel import Panel
from rich.table import Table
from rich import box


# ── Helpers ────────────────────────────────────────────────────────────────────

def embed_batch(texts: list, encoder, chunk_chars: int,
                encode_batch: int = 512) -> np.ndarray:
    """Embed a list of texts → (N, dim) array."""
    all_chunks, spans = [], []
    for t in texts:
        chunks = chunk_text(t, chunk_chars)
        spans.append(len(chunks))
        all_chunks.extend(chunks)
    if not all_chunks:
        return np.zeros((len(texts), encoder.get_sentence_embedding_dimension()))
    # OOM recovery: halve encode_batch until it fits, down to 4
    while True:
        try:
            import torch
            vecs = encoder.encode(
                all_chunks,
                batch_size=encode_batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            break
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" not in str(e).lower() or encode_batch <= 4:
                raise
            torch.cuda.empty_cache()
            encode_batch = max(4, encode_batch // 2)
            print(f"  ⚠ GPU OOM — retrying with encode_batch={encode_batch}",
                  file=sys.stderr)
    doc_vecs, i = [], 0
    for n in spans:
        doc_vecs.append(vecs[i:i + n].mean(axis=0))
        i += n
    return np.vstack(doc_vecs)


def embed_batch_dual(texts: list, encoders: list, chunk_chars: int,
                     encode_batch: int = 512) -> np.ndarray:
    """Embed using two GPU encoders sequentially — avoids CUDA threading issues.
    Encoder 0 handles the first half, encoder 1 the second half.
    Still faster than one GPU because both are warm and batch overhead is split.
    """
    if len(encoders) < 2 or len(texts) < 64:
        return embed_batch(texts, encoders[0], chunk_chars, encode_batch)
    half = len(texts) // 2
    v0 = embed_batch(texts[:half], encoders[0], chunk_chars, encode_batch)
    v1 = embed_batch(texts[half:], encoders[1], chunk_chars, encode_batch)
    return np.vstack([v0, v1])


def load_encoders(model_name: str, single_gpu: bool = False,
                  min_free_mb: int = 2500) -> list:
    """Load one encoder per GPU with enough free VRAM. Returns list of encoders."""
    import torch
    from sentence_transformers import SentenceTransformer
    encoders = []
    for i in range(torch.cuda.device_count()):
        if single_gpu and encoders:
            break
        try:
            free, _ = torch.cuda.mem_get_info(i)
            free_mb = free // (1024 * 1024)
            name = torch.cuda.get_device_name(i)
            if free_mb >= min_free_mb:
                console.print(
                    f"  [dim]GPU {i} ({name}): {free_mb:,} MB free — loading encoder[/]")
                enc = SentenceTransformer(model_name, device=f"cuda:{i}",
                                          trust_remote_code=True)
                encoders.append(enc)
            else:
                console.print(
                    f"  [dim]GPU {i} ({name}): {free_mb:,} MB free — skip[/]")
        except Exception as e:
            console.print(f"  [yellow]GPU {i}: {e}[/]")

    if not encoders:
        console.print("  [yellow]⚠ No GPU with sufficient VRAM — using CPU[/]")
        encoders = [SentenceTransformer(model_name, device="cpu",
                                         trust_remote_code=True)]
    elif len(encoders) > 1:
        console.print(f"  [green]✓ Dual-GPU embedding: {len(encoders)} encoders[/]")
    return encoders


def _embedding_worker(q, encoders, clf, classes, chunk_chars,
                      encode_batch, threshold, all_results, progress, task_emb):
    """Background thread: dequeue batches and embed them while extraction continues."""
    while True:
        batch = q.get()
        if batch is None:      # sentinel — extraction finished
            break
        _flush_batch(batch, encoders, clf, classes, chunk_chars,
                     encode_batch, threshold, all_results, progress, task_emb)
        q.task_done()


def safe_dest(dst_dir: Path, src: Path) -> Path:
    dst = dst_dir / src.name
    if not dst.exists():
        return dst
    stem, suffix = src.stem, src.suffix
    for n in range(1, 10000):
        candidate = dst_dir / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find free name for {src.name} in {dst_dir}")


def _flush_batch(texts_map: dict, encoders: list, clf, classes,
                 chunk_chars: int, encode_batch: int, threshold: float,
                 all_results: list, progress, task_emb):
    """Embed + predict one batch of extracted texts; append to all_results."""
    files_b = list(texts_map.keys())
    texts_b = list(texts_map.values())
    vecs    = embed_batch_dual(texts_b, encoders, chunk_chars, encode_batch)
    probas  = clf.predict_proba(vecs)
    for f, proba in zip(files_b, probas):
        order     = proba.argsort()[::-1]
        best_cls  = classes[order[0]]
        best_conf = float(proba[order[0]])
        sec_cls   = classes[order[1]] if len(order) > 1 else ""
        sec_conf  = float(proba[order[1]]) if len(order) > 1 else 0.0
        cls = best_cls if best_conf >= threshold else "_review"
        all_results.append((f, cls, best_conf, sec_cls, sec_conf))
    progress.advance(task_emb, advance=len(files_b))
    console.print(f"  [green]embedded {len(files_b):,}[/]  {_gpu_stat()}")


# ── Dual-GPU subprocess workers ────────────────────────────────────────────────

def _embed_proc_fn(in_q, out_q, embed_model, clf_path, chunk_chars,
                   encode_batch, threshold, device):
    """Subprocess worker: loads its own encoder+clf in an isolated heap.
    Heap isolation prevents the glibc corruption that occurs when two
    SentenceTransformer instances are loaded in the same process.
    Receives {Path: text} dicts, emits [(Path,cls,conf,sec_cls,sec_conf)] lists.
    """
    from sentence_transformers import SentenceTransformer

    try:
        encoder = SentenceTransformer(embed_model, device=device,
                                      trust_remote_code=True)
    except Exception as e:
        print(f"  ⚠ [{device}] encoder load failed: {e}", file=sys.stderr, flush=True)
        out_q.put(None)
        return

    try:
        bundle  = joblib.load(clf_path)
        clf     = bundle["clf"]
        classes = clf.classes_
    except Exception as e:
        print(f"  ⚠ [{device}] clf load failed: {e}", file=sys.stderr, flush=True)
        out_q.put(None)
        return

    print(f"  ✓ [{device}] embed subprocess ready", file=sys.stderr, flush=True)

    while True:
        batch = in_q.get()
        if batch is None:
            break
        try:
            files_b  = list(batch.keys())
            texts_b  = list(batch.values())
            vecs     = embed_batch(texts_b, encoder, chunk_chars, encode_batch)
            probas   = clf.predict_proba(vecs)
            rows = []
            for f, proba in zip(files_b, probas):
                order     = proba.argsort()[::-1]
                best_conf = float(proba[order[0]])
                cls       = classes[order[0]] if best_conf >= threshold else "_review"
                sec_cls   = classes[order[1]] if len(order) > 1 else ""
                sec_conf  = float(proba[order[1]]) if len(order) > 1 else 0.0
                rows.append((f, cls, best_conf, sec_cls, sec_conf))
            out_q.put(rows)
        except Exception as e:
            print(f"  ⚠ [{device}] batch error: {e}", file=sys.stderr, flush=True)
            out_q.put([])

    out_q.put(None)   # sentinel: subprocess finished


def _collect_results(out_qs, all_results, progress, task_embed):
    """Thread: drain result rows from embed subprocess output queues."""
    done = [False] * len(out_qs)
    while not all(done):
        for i, q in enumerate(out_qs):
            if done[i]:
                continue
            try:
                rows = q.get(timeout=0.05)
                if rows is None:
                    done[i] = True
                elif rows:
                    all_results.extend(rows)
                    progress.advance(task_embed, advance=len(rows))
            except queue.Empty:
                pass
            except Exception:
                pass


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="Directory of files to classify")
    parser.add_argument("-m", "--model", default=str(SCRIPT_DIR / "militia.joblib"),
                        help="Path to .joblib model bundle")
    parser.add_argument("-o", "--output", default="./sorted_output",
                        help="Output directory (created if absent)")
    parser.add_argument("--mode", choices=["copy", "symlink", "report"],
                        default="copy")
    parser.add_argument("--threshold", type=float, default=0.40,
                        help="Confidence below this → _review/ (default 0.40)")
    parser.add_argument("--workers", type=int,
                        default=min(12, max(1, (os.cpu_count() or 4) - 2)),
                        help="Parallel CPU extraction processes (default: min(12, cpu_count-2))")
    parser.add_argument("--maxtasks", type=int, default=50,
                        help="Pool maxtasksperchild — 50 avoids restarts within a 512-file window, 1=max isolation")
    parser.add_argument("--batch", type=int, default=512,
                        help="Documents per streaming window (default 512)")
    parser.add_argument("--encode-batch", type=int, default=512,
                        help="encoder.encode internal GPU batch size (default 512)")
    parser.add_argument("--chunk-chars", type=int, default=4000,
                        help="Characters per text chunk (default 4000)")
    parser.add_argument("--skip-ocr", action="store_true",
                        help="skip GPU OCR — image-only files go to _review/ (fast)")
    parser.add_argument("--max-ocr-pages", type=int, default=0,
                        help="limit OCR to first N pages (0=all)")
    parser.add_argument("--single-gpu", action="store_true", default=True,
                        help="use only the best single GPU (default: on; --no-single-gpu for dual)")
    parser.add_argument("--no-single-gpu", dest="single_gpu", action="store_false")
    args = parser.parse_args()

    source  = Path(args.source)
    output  = Path(args.output)
    model_p = Path(args.model)

    if not source.is_dir():
        console.print(f"[red]ERROR: source not a directory: {source}[/]")
        sys.exit(1)
    if not model_p.exists():
        console.print(f"[red]ERROR: model not found: {model_p}[/]")
        sys.exit(1)

    console.print(Panel(
        f"[bold white]sort_docs[/]  [dim]bulk classify & sort[/]\n\n"
        f"  source       [cyan]{source}[/]\n"
        f"  model        [cyan]{model_p.name}[/]\n"
        f"  output       [cyan]{output}[/]\n"
        f"  mode         [green]{args.mode}[/]  "
        f"threshold [yellow]{args.threshold:.0%}[/]\n"
        f"  workers      [green]{args.workers}[/]  "
        f"maxtasks [green]{args.maxtasks}[/]  "
        f"batch [green]{args.batch}[/]  "
        f"encode-batch [green]{args.encode_batch}[/]\n"
        f"  [dim]Ctrl-C → graceful shutdown (finishes current window, saves partial CSV)[/]",
        style="green", padding=(0, 2),
    ))

    bundle      = joblib.load(model_p)
    clf         = bundle["clf"]
    embed_model = bundle.get("embed_model", "BAAI/bge-m3")
    chunk_chars = bundle.get("chunk_chars", args.chunk_chars)
    classes     = clf.classes_

    # Check GPU count before deciding mode (torch.cuda.device_count doesn't
    # block spawn-based subprocesses since they get a fresh Python interpreter)
    import torch as _torch
    _n_gpus = _torch.cuda.device_count()
    if not args.single_gpu and _n_gpus < 2:
        console.print(f"[yellow]⚠ Only {_n_gpus} GPU(s) detected — single-GPU mode[/]")
        args.single_gpu = True

    if not args.single_gpu:
        console.print(
            "[bold cyan]▶ Dual-GPU subprocess mode[/] "
            "(cuda:0 + cuda:1, isolated heaps) ...")
        if not args.skip_ocr:
            console.print("  [yellow]⚠ OCR not supported in dual-GPU mode — "
                          "image-only files → _review/[/]")
        encoders   = []   # not used by main process in this mode
        ocr_reader = None
    else:
        console.print(f"\n[bold cyan]▶ Loading encoder[/] (single GPU) ...")
        encoders = load_encoders(embed_model, single_gpu=True)
        device   = str(encoders[0].device)
        ocr_reader = None if args.skip_ocr else get_ocr_reader(device)
        if args.skip_ocr:
            console.print("  [dim]--skip-ocr: GPU OCR disabled — image-only files → _review/[/]")

    console.print("\n[bold]Scanning source directory ...[/]")
    all_files = sorted(
        f for f in source.rglob("*")
        if f.is_file() and not f.name.startswith("._")
    )
    console.print(f"  Found [green]{len(all_files):,}[/] files\n")

    # ── True pipeline: extraction loop never blocks, embedding runs in background ─
    # Main thread drains imap_unordered pipe continuously (workers never stall).
    # Background thread dequeues batches and embeds them on GPU simultaneously.
    all_results   = []
    no_text_files = []
    pending_texts = {}
    t0_total = time.time()

    gpu_mode = "dual-GPU (subprocess)" if not args.single_gpu else "single-GPU (thread)"
    console.print(
        f"[bold]Streaming {len(all_files):,} files "
        f"({args.workers} workers · embed every {args.batch} · {gpu_mode})[/]\n")

    with _make_progress() as progress:
        task_extract = progress.add_task(
            f"[cyan]CPU extract[/] ({args.workers}w maxtasks={args.maxtasks})",
            total=len(all_files))
        task_embed = progress.add_task(
            f"[blue]GPU embed[/] (encode-batch={args.encode_batch})",
            total=len(all_files))

        if args.single_gpu:
            embed_q = queue.Queue()
            in_qs   = [embed_q]
            embed_thread = threading.Thread(
                target=_embedding_worker,
                args=(embed_q, encoders, clf, classes, chunk_chars,
                      args.encode_batch, args.threshold, all_results,
                      progress, task_embed),
                daemon=True,
            )
            embed_thread.start()
            procs     = []
            collector = embed_thread
        else:
            # Spawn two isolated embed processes — each loads its own encoder.
            # 'spawn' start method gives each subprocess a fresh Python heap,
            # avoiding the glibc corruption that occurs with two SentenceTransformer
            # instances in one process.
            ctx    = mp.get_context('spawn')
            in_qs  = [ctx.Queue(), ctx.Queue()]
            out_qs = [ctx.Queue(), ctx.Queue()]
            procs  = []
            for i in range(2):
                p = ctx.Process(
                    target=_embed_proc_fn,
                    args=(in_qs[i], out_qs[i], embed_model, str(model_p),
                          chunk_chars, args.encode_batch, args.threshold,
                          f"cuda:{i}"),
                    daemon=True,
                    name=f"embed-gpu{i}",
                )
                p.start()
                procs.append(p)
            console.print("  [dim]embed subprocesses spawned (cuda:0, cuda:1)[/]")
            collector = threading.Thread(
                target=_collect_results,
                args=(out_qs, all_results, progress, task_embed),
                daemon=True,
            )
            collector.start()

        # Restart loop: if a worker crashes (WorkerLostError propagates through
        # imap_unordered), pool.terminate() is called on __exit__ and other workers
        # get BrokenPipeError.  We restart a fresh pool for the remaining files.
        files_todo  = [str(f) for f in all_files]
        done_paths  = set()
        fail_counts = Counter()
        MAX_FAILS   = 2
        batch_idx   = 0   # round-robins across in_qs for dual-GPU load balancing

        while files_todo and not _shutdown.is_set():
            progressed = 0
            with Pool(processes=args.workers, maxtasksperchild=args.maxtasks) as pool:
                try:
                    for path_str, text in pool.imap_unordered(
                            _extract_cpu, files_todo, chunksize=1):
                        done_paths.add(path_str)
                        progressed += 1
                        if _shutdown.is_set():
                            pool.terminate()
                            break
                        f = Path(path_str)
                        progress.advance(task_extract)
                        if text.strip():
                            pending_texts[f] = text
                        else:
                            no_text_files.append(f)
                        if len(pending_texts) >= args.batch:
                            in_qs[batch_idx % len(in_qs)].put(dict(pending_texts))
                            pending_texts = {}
                            batch_idx += 1
                except Exception as exc:
                    console.print(
                        f"\n  [yellow]⚠ Worker crash ({type(exc).__name__}): {exc}[/]")

            files_todo = [p for p in files_todo if p not in done_paths]

            if files_todo and not _shutdown.is_set():
                if progressed == 0:
                    console.print(
                        f"  [red]⚠ No progress — giving up on "
                        f"{len(files_todo):,} remaining files[/]")
                    for p in files_todo:
                        no_text_files.append(Path(p))
                    break
                for p in files_todo:
                    fail_counts[p] += 1
                bad = [p for p in files_todo if fail_counts[p] >= MAX_FAILS]
                if bad:
                    for p in bad:
                        no_text_files.append(Path(p))
                        done_paths.add(p)
                    files_todo = [p for p in files_todo if fail_counts[p] < MAX_FAILS]
                    console.print(
                        f"  [yellow]Gave up on {len(bad):,} repeatedly-crashing files[/]")
                if files_todo:
                    console.print(
                        f"  [dim]Restarting pool: {len(files_todo):,} files remain[/]")

        # Final partial batch
        if pending_texts:
            in_qs[batch_idx % len(in_qs)].put(dict(pending_texts))

        # Shutdown embed thread / subprocess(es) and collector
        if args.single_gpu:
            in_qs[0].put(None)
            collector.join()
        else:
            for q in in_qs:
                q.put(None)          # sentinel to each embed subprocess
            for p in procs:
                p.join(timeout=300)
                if p.is_alive():
                    p.terminate()
            collector.join()         # wait for all results to be merged

    elapsed = time.time() - t0_total
    total_extracted = len(all_results)
    if _shutdown.is_set():
        console.print(Panel(
            f"[yellow]Partial run — {total_extracted:,} docs classified "
            f"({100*total_extracted/len(all_files):.0f}% of {len(all_files):,}).\n"
            f"Results saved to CSV. Rerun without interruption for full results.[/]",
            border_style="yellow",
        ))
    console.print(
        f"\n  [green]{total_extracted:,}[/] docs embedded in "
        f"[cyan]{elapsed/60:.1f} min[/]  "
        f"({total_extracted / max(elapsed, 1):.0f} docs/s)\n")

    # ── Build results list ─────────────────────────────────────────────────────
    results = [(f, "_review", 0.0, "", 0.0) for f in no_text_files]
    results.extend(all_results)  # _flush_batch already appended full tuples

    # ── Sort files ─────────────────────────────────────────────────────────────
    if args.mode != "report":
        output.mkdir(parents=True, exist_ok=True)
        with _make_progress() as progress:
            task_sort = progress.add_task(
                f"[white]Sorting ({args.mode})[/]", total=len(results))
            for f, cls, conf, _, _ in results:
                dst_dir = output / cls
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = safe_dest(dst_dir, f)
                try:
                    if args.mode == "copy":
                        shutil.copy2(f, dst)
                    elif args.mode == "symlink":
                        os.symlink(f.resolve(), dst)
                except Exception as exc:
                    console.print(f"  [yellow]⚠ {f.name}: {exc}[/]")
                progress.advance(task_sort)

    # ── CSV report ─────────────────────────────────────────────────────────────
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "sort_report.csv"
    with open(report_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "predicted_class", "confidence",
                    "2nd_class", "2nd_confidence", "source_path"])
        for f, cls, conf, sec_cls, sec_conf in results:
            w.writerow([f.name, cls, f"{conf:.3f}", sec_cls,
                        f"{sec_conf:.3f}", str(f)])

    # ── Summary ────────────────────────────────────────────────────────────────
    counts = Counter(cls for _, cls, _, _, _ in results)
    t = Table(title="[bold green]Sort summary[/]", box=box.SIMPLE,
              border_style="green")
    t.add_column("Class", style="cyan")
    t.add_column("Files", justify="right", style="green")
    t.add_column("Bar", style="dim")
    total = len(results)
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        bar   = "█" * int(30 * n / max(counts.values()))
        pct   = f"{100*n/total:.1f}%"
        label = f"[yellow]{cls}[/]" if cls == "_review" else cls
        t.add_row(label, f"{n:,} ({pct})", f"[dim]{bar}[/]")
    t.add_row("[bold]TOTAL[/]", f"[bold green]{total:,}[/]", "")
    console.print(t)
    console.print(f"\n  Report → [cyan]{report_path}[/]")
    if args.mode != "report":
        console.print(f"  Sorted → [cyan]{output}[/]")


if __name__ == "__main__":
    main()

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
        --encode-batch 512     # encoder.encode internal GPU batch size
        --skip-ocr             # skip GPU OCR (fast; image files → _review/)
        --single-gpu           # disable dual-GPU embedding

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
import os
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
    vecs = encoder.encode(
        all_chunks,
        batch_size=encode_batch,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
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
    parser.add_argument("--single-gpu", action="store_true",
                        help="disable dual-GPU embedding (use only the best single GPU)")
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

    console.print(f"\n[bold cyan]▶ Loading encoder(s)[/] — dual GPU: {not args.single_gpu} ...")
    encoders = load_encoders(embed_model, single_gpu=args.single_gpu)

    device     = str(encoders[0].device)
    ocr_reader = None if args.skip_ocr else get_ocr_reader(device)
    if args.skip_ocr:
        console.print("  [dim]--skip-ocr: GPU OCR disabled — image-only files → _review/[/]")

    console.print("\n[bold]Scanning source directory ...[/]")
    all_files = sorted(
        f for f in source.rglob("*")
        if f.is_file() and not f.name.startswith("._")
    )
    console.print(f"  Found [green]{len(all_files):,}[/] files\n")

    # ── Persistent Pool + streaming embed: workers run continuously ───────────
    # One pool for all files; embed every --batch extracted texts immediately.
    # CPU workers and GPU encoder both active simultaneously — no idle gaps.
    all_results   = []
    no_text_files = []
    pending_texts = {}   # Path → text, accumulating toward next embed batch
    t0_total = time.time()

    console.print(
        f"[bold]Streaming {len(all_files):,} files "
        f"({args.workers} workers · embed every {args.batch}) ...[/]\n")

    with _make_progress() as progress:
        task_extract = progress.add_task(
            f"[cyan]CPU extract[/] ({args.workers}w maxtasks={args.maxtasks})",
            total=len(all_files))
        task_embed = progress.add_task(
            f"[blue]GPU embed[/] (encode-batch={args.encode_batch})",
            total=len(all_files))

        with Pool(processes=args.workers, maxtasksperchild=args.maxtasks) as pool:
            try:
                result_iter = pool.imap_unordered(
                    _extract_cpu, [str(f) for f in all_files], chunksize=1)
                for path_str, text in result_iter:
                    if _shutdown.is_set():
                        pool.terminate()
                        break
                    f = Path(path_str)
                    progress.advance(task_extract)
                    if text.strip():
                        pending_texts[f] = text
                    else:
                        no_text_files.append(f)

                    # GPU embeds whenever a full batch is ready
                    # Workers keep running during this call — true overlap
                    if len(pending_texts) >= args.batch:
                        _flush_batch(pending_texts, encoders, clf, classes,
                                     chunk_chars, args.encode_batch,
                                     args.threshold, all_results,
                                     progress, task_embed)
                        pending_texts = {}
            except Exception as exc:
                console.print(f"  [yellow]⚠ Pool error: {exc}[/]")

        # Final partial batch (remaining texts after last full flush)
        if pending_texts:
            _flush_batch(pending_texts, encoders, clf, classes,
                         chunk_chars, args.encode_batch,
                         args.threshold, all_results, progress, task_embed)

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

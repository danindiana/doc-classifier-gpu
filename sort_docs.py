#!/usr/bin/env python3
"""Bulk-classify and sort a document collection using a trained militia.joblib model.

Usage:
    CUDA_VISIBLE_DEVICES=0 python sort_docs.py SOURCE_DIR \\
        --model militia.joblib \\
        --output ./sorted_output \\
        --mode copy        # copy | symlink | report
        --threshold 0.40   # confidence < threshold → _review/
        --workers 14       # parallel CPU extraction processes
        --batch 256        # documents per embedding batch

Output:
    sorted_output/
    ├── strategy/          ← predicted class ≥ threshold
    ├── medical/
    ├── ...
    ├── _review/           ← max confidence < threshold
    └── sort_report.csv    ← full results table
"""

import argparse
import csv
import os
import shutil
import sys
import time
from collections import Counter
from multiprocessing.pool import Pool
from pathlib import Path

import joblib
import numpy as np

# ── Bootstrap: add script dir to path so we can import from doc_classifier_gpu
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

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

def embed_batch(texts: list, encoder, chunk_chars: int) -> np.ndarray:
    """Embed a list of texts, return (N, 1024) array."""
    all_chunks, spans = [], []
    for t in texts:
        chunks = chunk_text(t, chunk_chars)
        spans.append(len(chunks))
        all_chunks.extend(chunks)
    if not all_chunks:
        return np.zeros((len(texts), encoder.get_sentence_embedding_dimension()))
    vecs = encoder.encode(
        all_chunks,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    doc_vecs, i = [], 0
    for n in spans:
        doc_vecs.append(vecs[i:i + n].mean(axis=0))
        i += n
    return np.vstack(doc_vecs)


def safe_dest(dst_dir: Path, src: Path) -> Path:
    """Return a non-colliding destination path inside dst_dir."""
    dst = dst_dir / src.name
    if not dst.exists():
        return dst
    stem, suffix = src.stem, src.suffix
    for n in range(1, 10000):
        candidate = dst_dir / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find free name for {src.name} in {dst_dir}")


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
                        default="copy",
                        help="copy=copy files, symlink=create symlinks, report=CSV only")
    parser.add_argument("--threshold", type=float, default=0.40,
                        help="Confidence below this → _review/ (default 0.40)")
    parser.add_argument("--workers", type=int,
                        default=min(16, max(1, (os.cpu_count() or 4) - 2)),
                        help="Parallel CPU extraction processes")
    parser.add_argument("--batch", type=int, default=256,
                        help="Documents per embedding batch (default 256)")
    parser.add_argument("--chunk-chars", type=int, default=4000,
                        help="Characters per text chunk (default 4000)")
    parser.add_argument("--skip-ocr", action="store_true",
                        help="skip GPU OCR — image-only files go to _review/ (fast)")
    parser.add_argument("--max-ocr-pages", type=int, default=0,
                        help="limit OCR to first N pages per file (0=all, try 3-5 for speed)")
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

    # ── Load model ────────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold white]sort_docs[/]  [dim]bulk classify & sort[/]\n\n"
        f"  source    [cyan]{source}[/]\n"
        f"  model     [cyan]{model_p.name}[/]\n"
        f"  output    [cyan]{output}[/]\n"
        f"  mode      [green]{args.mode}[/]  "
        f"  threshold [yellow]{args.threshold:.0%}[/]  "
        f"  workers   [green]{args.workers}[/]",
        style="green", padding=(0, 2),
    ))

    bundle = joblib.load(model_p)
    clf         = bundle["clf"]
    embed_model = bundle.get("embed_model", "BAAI/bge-m3")
    chunk_chars = bundle.get("chunk_chars", args.chunk_chars)
    classes     = clf.classes_

    encoder    = get_encoder(embed_model)
    device     = str(encoder.device)
    ocr_reader = None if args.skip_ocr else get_ocr_reader(device)
    if args.skip_ocr:
        console.print("  [dim]--skip-ocr: GPU OCR disabled — image-only files → _review/[/]")
    elif args.max_ocr_pages:
        console.print(f"  [dim]--max-ocr-pages {args.max_ocr_pages}: OCR limited to first {args.max_ocr_pages} pages[/]")

    # ── Scan files ────────────────────────────────────────────────────────────
    console.print("\n[bold]Scanning source directory ...[/]")
    all_files = sorted(
        f for f in source.rglob("*")
        if f.is_file() and not f.name.startswith("._")
    )
    console.print(f"  Found [green]{len(all_files):,}[/] files\n")

    # ── Phase A: parallel CPU extraction ─────────────────────────────────────
    texts_map   = {}   # Path → text string
    needs_ocr   = []
    needs_seq   = []

    with _make_progress() as progress:
        task = progress.add_task(
            f"[cyan]CPU extract[/] ({args.workers} workers)", total=len(all_files))
        with Pool(processes=args.workers, maxtasksperchild=1) as pool:
            async_results = [(f, pool.apply_async(_extract_cpu, (str(f),)))
                             for f in all_files]
            for f, ar in async_results:
                try:
                    _, text = ar.get(timeout=60)
                    if text.strip():
                        texts_map[f] = text
                    else:
                        needs_ocr.append(f)
                except Exception:
                    needs_seq.append(f)
                progress.advance(task)

        # Phase A fallback
        if needs_seq:
            task_seq = progress.add_task(
                "[yellow]sequential fallback[/]", total=len(needs_seq))
            for f in needs_seq:
                text = extract_text(f, ocr_reader=None)
                if text.strip():
                    texts_map[f] = text
                else:
                    needs_ocr.append(f)
                progress.advance(task_seq)

        # Phase B: GPU OCR (skipped if --skip-ocr)
        if needs_ocr and ocr_reader is not None:
            task_ocr = progress.add_task(
                f"[magenta]GPU OCR[/]"
                + (f" [dim](max {args.max_ocr_pages} pages)[/]" if args.max_ocr_pages else ""),
                total=len(needs_ocr))
            for f in sorted(needs_ocr):
                text = extract_text(f, ocr_reader,
                                    max_ocr_pages=args.max_ocr_pages)
                if text.strip():
                    texts_map[f] = text
                progress.advance(task_ocr)

    console.print(f"  Extracted text from [green]{len(texts_map):,}[/] / "
                  f"{len(all_files):,} files  "
                  f"([dim]{len(all_files)-len(texts_map):,} no extractable text → _review[/])\n")

    # ── Phase C: batch embedding + prediction ─────────────────────────────────
    extracted_files = list(texts_map.keys())
    extracted_texts = [texts_map[f] for f in extracted_files]
    n_batches = (len(extracted_files) + args.batch - 1) // args.batch

    all_probas = []
    t0_embed = time.time()

    with _make_progress() as progress:
        task_emb = progress.add_task(
            f"[blue]Embedding[/] (batch={args.batch})", total=n_batches)
        for i in range(n_batches):
            batch_texts = extracted_texts[i*args.batch:(i+1)*args.batch]
            vecs = embed_batch(batch_texts, encoder, chunk_chars)
            all_probas.append(clf.predict_proba(vecs))
            gpu_str = _gpu_stat()
            console.print(
                f"  batch {i+1}/{n_batches}  "
                f"[green]{len(batch_texts)}[/] docs  {gpu_str}")
            progress.advance(task_emb)

    embed_elapsed = time.time() - t0_embed
    all_probas = np.vstack(all_probas)
    console.print(f"  Embedded [green]{len(extracted_files):,}[/] docs "
                  f"in [cyan]{embed_elapsed/60:.1f} min[/]\n")

    # ── Phase D: sort files ───────────────────────────────────────────────────
    results = []   # (Path, class_name, confidence, 2nd_class, 2nd_conf)

    # Files with no text → _review
    for f in all_files:
        if f not in texts_map:
            results.append((f, "_review", 0.0, "", 0.0))

    for f, proba in zip(extracted_files, all_probas):
        order = proba.argsort()[::-1]
        best_cls  = classes[order[0]]
        best_conf = float(proba[order[0]])
        sec_cls   = classes[order[1]] if len(order) > 1 else ""
        sec_conf  = float(proba[order[1]]) if len(order) > 1 else 0.0
        cls = best_cls if best_conf >= args.threshold else "_review"
        results.append((f, cls, best_conf, sec_cls, sec_conf))

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

    # ── Write CSV report ──────────────────────────────────────────────────────
    report_path = (output / "sort_report.csv") if args.mode != "report" \
                  else Path("sort_report.csv")
    if args.mode == "report":
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / "sort_report.csv"

    output.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "predicted_class", "confidence",
                    "2nd_class", "2nd_confidence", "source_path"])
        for f, cls, conf, sec_cls, sec_conf in results:
            w.writerow([f.name, cls, f"{conf:.3f}", sec_cls,
                        f"{sec_conf:.3f}", str(f)])

    # ── Summary table ─────────────────────────────────────────────────────────
    counts = Counter(cls for _, cls, _, _, _ in results)
    t = Table(title="[bold green]Sort summary[/]", box=box.SIMPLE,
              border_style="green", show_lines=False)
    t.add_column("Class", style="cyan")
    t.add_column("Files", justify="right", style="green")
    t.add_column("Bar", style="dim")
    total = len(results)
    bar_max = 30
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(bar_max * n / max(counts.values()))
        pct = f"{100*n/total:.1f}%"
        label = f"[yellow]{cls}[/]" if cls == "_review" else cls
        t.add_row(label, f"{n:,} ({pct})", f"[dim]{bar}[/]")
    t.add_row("[bold]TOTAL[/]", f"[bold green]{total:,}[/]", "")
    console.print(t)
    console.print(f"\n  Report → [cyan]{report_path}[/]")
    if args.mode != "report":
        console.print(f"  Sorted → [cyan]{output}[/]")


if __name__ == "__main__":
    main()

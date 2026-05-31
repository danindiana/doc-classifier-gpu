#!/usr/bin/env python3
"""Folder-driven document-type classifier (GPU embedding version).

Each immediate sub-folder of the training directory is a class label;
the files inside it (recursively) are that class's training examples.

Pipeline:
  - CPU text extraction runs in parallel across processes (ProcessPoolExecutor)
  - Scanned/image PDFs and image files fall back to GPU OCR (EasyOCR, sequential)
  - Text is chunked, embedded on GPU (sentence-transformers), mean-pooled per doc
  - LogisticRegression classifier trained on the embeddings
  - rich TUI: progress bars, GPU stats, styled output, summary table

Usage:
    CUDA_VISIBLE_DEVICES=0 \
    python doc_classifier_gpu.py train  /path/to/training_folder -m model.joblib
    python doc_classifier_gpu.py predict /path/to/file_or_folder  -m model.joblib

Options worth knowing:
    --embed-model   sentence-transformers model (default: BAAI/bge-m3).
                    Lighter: BAAI/bge-small-en-v1.5 or all-MiniLM-L6-v2.
    --chunk-chars   characters per chunk before pooling (default 4000).
    --workers       parallel CPU extraction processes (default: cpu_count-2, max 16).
"""

import argparse
import os
import subprocess
import sys
import time
from collections import Counter
from multiprocessing.pool import Pool
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console(highlight=False)

def _pick_device(min_free_mb: int = 2500) -> str:
    """Return the first CUDA device with >= min_free_mb free VRAM, or 'cpu'."""
    import torch
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
            free_mb = free // (1024 * 1024)
            name = torch.cuda.get_device_name(i)
            if free_mb >= min_free_mb:
                console.print(
                    f"  [dim]GPU {i} ({name}): {free_mb:,} MB free — selected[/]")
                return f"cuda:{i}"
            console.print(
                f"  [dim]GPU {i} ({name}): {free_mb:,} MB free — skip (need {min_free_mb:,})[/]")
        except Exception:
            pass
    console.print("  [yellow]⚠ No GPU with sufficient free VRAM — falling back to CPU[/]")
    return "cpu"


TEXT_SUFFIXES = {
    ".txt", ".md", ".rst", ".csv", ".tsv", ".log",
    ".json", ".xml", ".html", ".htm", ".eml",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}


def _gpu_stat(gpu_id: int = 0) -> str:
    """Return a styled one-line GPU stat string for console.print()."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_id}",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        ).strip()
        util, used, total, temp = [x.strip() for x in out.split(",")]
        bar_w = 14
        filled = int(bar_w * int(util) / 100)
        bar = "█" * filled + "░" * (bar_w - filled)
        return (f"[bold green]GPU{gpu_id}[/] [green]{bar}[/] "
                f"[green]{util:>3}%[/]  "
                f"[cyan]{used}/{total} MiB[/]  "
                f"[yellow]{temp}°C[/]")
    except Exception:
        return "[dim]GPU stat unavailable[/]"


def _make_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def _ocr_pdf(path: Path, ocr_reader, max_pages: int = 0) -> str:
    """Render each page of an image PDF and OCR it on GPU.
    max_pages=0 means all pages; set e.g. 3 for fast bulk classification.
    """
    import fitz  # pymupdf
    doc = fitz.open(str(path))
    page_list = list(doc)
    if max_pages:
        page_list = page_list[:max_pages]
    pages = []
    for page in page_list:
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        result = ocr_reader.readtext(img, detail=0)
        pages.append(" ".join(result))
    doc.close()
    return "\n".join(pages)


def extract_text(path: Path, ocr_reader=None, max_ocr_pages: int = 0) -> str:
    """Return the text content of a file, or '' if it can't be read.
    max_ocr_pages: limit GPU OCR to first N pages (0=unlimited).
    """
    suffix = path.suffix.lower()
    try:
        if suffix in TEXT_SUFFIXES:
            return path.read_text(errors="ignore")
        if suffix == ".pdf":
            import fitz  # pymupdf — no Pillow dependency, handles corrupt images safely
            try:
                doc = fitz.open(str(path))
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
            except Exception as exc:
                print(f"  ! fitz error {path.name}: {exc}", file=sys.stderr)
                return ""
            if text.strip():
                return text
            if ocr_reader is not None:
                return _ocr_pdf(path, ocr_reader, max_pages=max_ocr_pages)
            return ""
        if suffix in IMAGE_SUFFIXES:
            if ocr_reader is not None:
                result = ocr_reader.readtext(str(path), detail=0)
                return " ".join(result)
            return ""
        if suffix == ".docx":
            import docx
            return "\n".join(p.text for p in docx.Document(path).paragraphs)
        if suffix == ".rtf":
            from striprtf.striprtf import rtf_to_text
            return rtf_to_text(path.read_text(errors="ignore"))
    except Exception as exc:
        print(f"  ! skipping {path.name}: {exc}", file=sys.stderr)
    return ""


def _extract_cpu(path_str: str) -> tuple:
    """Worker: CPU-only extraction, no OCR. Returns (path_str, text)."""
    return path_str, extract_text(Path(path_str), ocr_reader=None)


def chunk_text(text: str, chunk_chars: int, overlap: int = 200):
    """Split text into overlapping chunks so long docs are fully represented."""
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_chars])
        start += chunk_chars - overlap
    return chunks


def load_dataset(root: Path):
    """Walk sub-folders; return (texts, labels, paths). CPU only, no parallelism."""
    texts, labels, paths = [], [], []
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not class_dirs:
        sys.exit(f"No sub-folders in {root}; nothing to use as labels.")
    for label_dir in class_dirs:
        class_count = 0
        for f in sorted(label_dir.rglob("*")):
            if f.is_file():
                console.print(f"  [[cyan]{label_dir.name}[/]] [dim]{f.name}[/] ...",
                               end=" ")
                text = extract_text(f)
                if text.strip():
                    texts.append(text)
                    labels.append(label_dir.name)
                    paths.append(f)
                    console.print(f"[green]{len(text):,} chars[/]")
                    class_count += 1
                else:
                    console.print("[dim](skipped)[/]")
        console.print(
            f"  [bold]→[/] [cyan]{label_dir.name}[/]: [green]{class_count}[/] docs\n")
    return texts, labels, paths


def get_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    import torch
    device = _pick_device(min_free_mb=2500)
    console.print(
        f"\n[bold cyan]▶ Loading embedding model[/] [white]'{model_name}'[/]"
        f" on [green]{device}[/] ...")
    try:
        return SentenceTransformer(model_name, device=device, trust_remote_code=True)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device != "cpu":
            console.print(f"  [yellow]⚠ OOM on {device} — retrying on CPU[/]")
            torch.cuda.empty_cache()
            return SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
        raise


def get_ocr_reader(device: str):
    import easyocr
    gpu = "cuda" in device
    label = device if gpu else "CPU"
    console.print(
        f"[bold cyan]▶ Loading OCR model[/] (easyocr) on [green]{label}[/] ...")
    try:
        return easyocr.Reader(['en'], gpu=gpu, verbose=False)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and gpu:
            console.print("  [yellow]⚠ OOM loading EasyOCR on GPU — falling back to CPU[/]")
            return easyocr.Reader(['en'], gpu=False, verbose=False)
        raise


def embed_documents(texts, encoder, chunk_chars: int,
                    encode_batch: int = 512) -> np.ndarray:
    """Embed each document as the mean of its chunk embeddings."""
    all_chunks, spans = [], []
    for t in texts:
        chunks = chunk_text(t, chunk_chars)
        spans.append(len(chunks))
        all_chunks.extend(chunks)

    chunk_vecs = encoder.encode(
        all_chunks,
        batch_size=encode_batch,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    doc_vecs, i = [], 0
    for n in spans:
        doc_vecs.append(chunk_vecs[i:i + n].mean(axis=0))
        i += n
    return np.vstack(doc_vecs)


def _extract_class(label_dir: Path, workers: int, ocr_reader,
                   maxtasks: int = 20) -> list:
    """Extract all text from one class directory. Returns list of text strings."""
    files = sorted(f for f in label_dir.rglob("*")
                   if f.is_file() and not f.name.startswith("._"))
    if not files:
        return []

    ext = label_dir.name
    console.print(Panel(
        f"[bold white]{ext}[/]  "
        f"[dim]{len(files)} files · {workers} CPU workers[/]",
        style="blue", expand=False, padding=(0, 1),
    ))

    cpu_texts, needs_ocr, needs_sequential = [], [], []

    # Phase A: parallel CPU extraction — maxtasksperchild=1 isolates crashes per file
    with _make_progress() as progress:
        task_cpu = progress.add_task(
            f"[cyan]CPU extract[/] [dim]{ext}[/]", total=len(files))
        with Pool(processes=workers, maxtasksperchild=maxtasks) as pool:
            async_results = [(f, pool.apply_async(_extract_cpu, (str(f),)))
                             for f in files]
            for f, ar in async_results:
                try:
                    path_str, text = ar.get(timeout=60)
                    if text.strip():
                        console.print(
                            f"  [dim]{f.name[:40]}[/] "
                            f"[green]{len(text):,}c[/]")
                        cpu_texts.append(text)
                    else:
                        needs_ocr.append(f)
                except Exception as exc:
                    console.print(
                        f"  [yellow]⚠ {f.name[:32]}: worker error — retrying[/]")
                    needs_sequential.append(f)
                progress.advance(task_cpu)

        # Phase B: sequential fallback
        if needs_sequential:
            task_seq = progress.add_task(
                "[yellow]sequential fallback[/]", total=len(needs_sequential))
            for f in needs_sequential:
                console.print(f"  [dim]{f.name[:40]}[/] [yellow](fallback)[/]",
                               end=" ")
                text = extract_text(f, ocr_reader=None)
                if text.strip():
                    cpu_texts.append(text)
                    console.print(f"[green]{len(text):,}c[/]")
                else:
                    needs_ocr.append(f)
                    console.print("[dim]→ OCR queue[/]")
                progress.advance(task_seq)

        # Phase C: GPU OCR
        if needs_ocr:
            task_ocr = progress.add_task(
                "[magenta]GPU OCR[/]", total=len(needs_ocr))
            ocr_texts = []
            for f in sorted(needs_ocr):
                console.print(f"  [dim]{f.name[:40]}[/] [magenta](OCR)[/]",
                               end=" ")
                text = extract_text(f, ocr_reader)
                if text.strip():
                    ocr_texts.append(text)
                    console.print(f"[green]{len(text):,}c[/]")
                else:
                    console.print("[dim](skipped)[/]")
                progress.advance(task_ocr)
        else:
            ocr_texts = []

    n_docs = len(cpu_texts) + len(ocr_texts)
    gpu_str = _gpu_stat()
    console.print(
        f"  [bold green]✓[/] [cyan]{ext}[/]: "
        f"[green]{n_docs} docs[/]  {gpu_str}\n")

    return cpu_texts + ocr_texts


def train(args):
    import torch
    root = Path(args.folder)
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not class_dirs:
        sys.exit(f"No sub-folders in {root}; nothing to use as labels.")

    console.print(Panel(
        f"[bold white]doc_classifier_gpu — Training[/]\n"
        f"[dim]{len(class_dirs)} classes · {args.embed_model} · "
        f"{args.workers} workers · chunk {args.chunk_chars}c[/]",
        style="green", padding=(0, 2),
    ))

    encoder = get_encoder(args.embed_model)
    device = str(encoder.device)
    ocr_reader = get_ocr_reader(device)

    all_X, all_y = [], []
    counts = Counter()
    t0_total = time.time()

    with _make_progress() as overall:
        task_all = overall.add_task(
            "[white]overall classes[/]", total=len(class_dirs))

        for label_dir in class_dirs:
            texts = _extract_class(label_dir, args.workers, ocr_reader, maxtasks=args.maxtasks)

            if not texts:
                console.print(
                    f"  [dim]→ {label_dir.name}: no readable documents, skipping[/]\n")
                overall.advance(task_all)
                continue

            t0 = time.time()
            console.print(
                f"  [bold cyan]▶ Embedding[/] [cyan]{label_dir.name}[/] "
                f"([green]{len(texts)}[/] docs) ...")
            X = embed_documents(texts, encoder, args.chunk_chars, encode_batch=args.encode_batch)
            elapsed = time.time() - t0

            all_X.append(X)
            all_y.extend([label_dir.name] * len(texts))
            counts[label_dir.name] = len(texts)

            gpu_str = _gpu_stat()
            console.print(
                f"  [green]✓[/] [cyan]{label_dir.name}[/] embedded "
                f"in [green]{elapsed:.1f}s[/]  {gpu_str}\n")
            overall.advance(task_all)

    if not all_X:
        sys.exit("No readable documents found.")

    X = np.vstack(all_X)
    y = np.array(all_y)

    # Summary table
    t = Table(title="[bold green]Documents loaded[/]", style="cyan",
              border_style="green", show_lines=False)
    t.add_column("Class", style="cyan", no_wrap=True)
    t.add_column("Docs", justify="right", style="green")
    for label, n in sorted(counts.items()):
        t.add_row(label, str(n))
    t.add_row("[bold]TOTAL[/]", f"[bold green]{len(y)}[/]")
    console.print(t)

    clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=10.0)

    min_count = min(counts.values())
    if len(counts) >= 2 and min_count >= 2:
        folds = min(5, min_count)
        console.print(f"\n[bold]Running {folds}-fold cross-validation ...[/]")
        scores = cross_val_score(clf, X, y, cv=folds)
        console.print(
            f"[bold green]{folds}-fold CV accuracy: "
            f"{scores.mean():.3f} ± {scores.std():.3f}[/]")
    else:
        console.print(
            "\n[dim](Too few examples per class for cross-validation.)[/]")

    clf.fit(X, y)
    joblib.dump(
        {"clf": clf, "embed_model": args.embed_model,
         "chunk_chars": args.chunk_chars},
        args.model,
    )

    total_time = time.time() - t0_total
    console.print(Panel(
        f"[bold green]✓ Saved → {args.model}[/]\n"
        f"[dim]Total time: {total_time/60:.1f} min · "
        f"{len(y)} docs · {len(counts)} classes[/]",
        style="green", padding=(0, 2),
    ))


def predict(args):
    bundle = joblib.load(args.model)
    clf = bundle["clf"]
    encoder = get_encoder(bundle["embed_model"])
    chunk_chars = bundle["chunk_chars"]
    device = str(encoder.device)
    ocr_reader = get_ocr_reader(device)

    target = Path(args.path)
    files = sorted([target] if target.is_file() else
                   (f for f in target.rglob("*")
                    if f.is_file() and not f.name.startswith("._")))
    if not files:
        sys.exit(f"No files found at {target}")

    console.print(f"\n[bold cyan]▶ Classifying[/] {len(files)} file(s) ...")

    cpu_results, needs_ocr, needs_sequential = [], [], []
    with Pool(processes=args.workers, maxtasksperchild=args.maxtasks) as pool:
        async_results = [(f, pool.apply_async(_extract_cpu, (str(f),)))
                         for f in files]
        for f, ar in async_results:
            try:
                path_str, text = ar.get(timeout=60)
                if text.strip():
                    cpu_results.append((f, text))
                else:
                    needs_ocr.append(f)
            except Exception:
                needs_sequential.append(f)

    for f in needs_sequential:
        text = extract_text(f, ocr_reader=None)
        if text.strip():
            cpu_results.append((f, text))
        else:
            needs_ocr.append(f)

    ocr_results = []
    for f in sorted(needs_ocr):
        text = extract_text(f, ocr_reader)
        if text.strip():
            ocr_results.append((f, text))
        else:
            console.print(f"[dim]{f.name:<40} <no extractable text>[/]")

    all_results = cpu_results + ocr_results
    if not all_results:
        return

    valid_files, texts = zip(*all_results)
    X = embed_documents(list(texts), encoder, chunk_chars, encode_batch=getattr(args, "encode_batch", 512))
    probas = clf.predict_proba(X)
    classes = clf.classes_

    t = Table(style="cyan", border_style="blue", show_lines=False)
    t.add_column("File", style="white", no_wrap=True)
    t.add_column("1st", style="green")
    t.add_column("2nd", style="cyan")
    t.add_column("3rd", style="dim")

    for f, proba in zip(valid_files, probas):
        order = proba.argsort()[::-1]
        top3 = [(classes[i], proba[i]) for i in order[:3]]
        t.add_row(
            f.name[:40],
            f"{top3[0][0]} {top3[0][1]:.0%}",
            f"{top3[1][0]} {top3[1][1]:.0%}" if len(top3) > 1 else "",
            f"{top3[2][0]} {top3[2][1]:.0%}" if len(top3) > 2 else "",
        )

    console.print(t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-m", "--model", default="model.joblib")
    common.add_argument("--embed-model", default="BAAI/bge-m3")
    common.add_argument("--chunk-chars", type=int, default=4000)
    common.add_argument(
        "--workers", type=int,
        default=min(24, max(1, (os.cpu_count() or 4) - 2)),
        help="parallel CPU extraction processes (default: cpu_count-2, max 24)",
    )
    common.add_argument(
        "--maxtasks", type=int, default=20,
        help="pool maxtasksperchild — 20 cuts fork overhead, 1=max isolation",
    )
    common.add_argument(
        "--encode-batch", type=int, default=512,
        help="encoder.encode internal GPU batch size (default 512)",
    )

    p_train = sub.add_parser("train", parents=[common],
                             help="train from a labelled folder")
    p_train.add_argument("folder")
    p_train.set_defaults(func=train)

    p_pred = sub.add_parser("predict", parents=[common],
                            help="classify a file or folder")
    p_pred.add_argument("path")
    p_pred.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

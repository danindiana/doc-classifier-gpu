#!/usr/bin/env python3
"""
doc_classifier_gpu — Training-to-Inference Wizard  v2.0.0

Interactive guide covering the full pipeline:
  environment check → corpus inspection → training → model inspection
  → inference (single / folder) → bulk sort (sort_docs.py)

GPU selection is automatic — the script probes free VRAM on all GPUs and
picks the best one. No CUDA_VISIBLE_DEVICES needed.

Run with the venv active:
    source ~/doc-clf-gpu-env/bin/activate
    python wizard.py

Or directly (wizard resolves the venv automatically):
    python3 wizard.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# ── Resolve paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).parent.resolve()
CLASSIFIER     = SCRIPT_DIR / "doc_classifier_gpu.py"
SORTER         = SCRIPT_DIR / "sort_docs.py"
DEFAULT_CORPUS = Path.home() / "Documents/claude_creations/2026-05-28_militia-copy"
DEFAULT_MODEL  = SCRIPT_DIR / "militia.joblib"
VENV_PYTHON    = Path.home() / "doc-clf-gpu-env/bin/python"
VENV_DIR       = Path.home() / "doc-clf-gpu-env"

PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# ── Bootstrap rich ─────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, IntPrompt, Prompt
    from rich.rule import Rule
    from rich.table import Table
    from rich import box
except ImportError:
    print("ERROR: 'rich' not found. Activate the venv first:")
    print("  source ~/doc-clf-gpu-env/bin/activate")
    sys.exit(1)

console = Console(highlight=False)

VERSION = "2.0.0"
EMBED_MODELS = {
    "1": ("BAAI/bge-m3",            "Best accuracy · 8192-token ctx · 2.2 GB · multilingual ← recommended"),
    "2": ("BAAI/bge-small-en-v1.5", "Fast · 512-token ctx · 130 MB · English only"),
    "3": ("all-MiniLM-L6-v2",       "Tiny · 256-token ctx · 90 MB  · English only"),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def banner():
    console.print(Panel.fit(
        f"[bold white]doc_classifier_gpu[/]  [dim]v{VERSION}[/]\n"
        f"[dim]Training-to-Inference Wizard[/]\n\n"
        f"[cyan]GPU embedding document classifier[/]\n"
        f"[dim]bge-m3 · EasyOCR · PyMuPDF · LogisticRegression[/]\n"
        f"[dim]GPU auto-selected by free VRAM — no CUDA_VISIBLE_DEVICES needed[/]",
        border_style="green", padding=(1, 4),
    ))
    console.print()


def section(title: str):
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/]", style="cyan"))
    console.print()


def ok(msg):   console.print(f"  [bold green]✓[/]  {msg}")
def warn(msg): console.print(f"  [bold yellow]⚠[/]  {msg}")
def err(msg):  console.print(f"  [bold red]✗[/]  {msg}")
def info(msg): console.print(f"  [dim]→[/]  {msg}")


def all_gpu_stats():
    """Return list of (id, name, util, free_mb, total_mb, temp) for all GPUs."""
    results = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.free,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 6:
                results.append((int(parts[0]), parts[1], int(parts[2]),
                                 int(parts[3]), int(parts[4]), int(parts[5])))
    except Exception:
        pass
    return results


# ── Step 1: Environment check ──────────────────────────────────────────────────

def check_env() -> bool:
    section("Step 1 — Environment Check")
    all_ok = True

    # venv
    if VENV_DIR.exists():
        ok(f"venv at [cyan]{VENV_DIR}[/]")
    else:
        err(f"venv missing: [cyan]{VENV_DIR}[/]")
        info("See README §Blackwell PyTorch install to create it.")
        all_ok = False

    # torch + CUDA
    try:
        result = subprocess.check_output(
            [PYTHON, "-c",
             "import torch; "
             "print(torch.__version__); "
             "print(torch.cuda.device_count()); "
             "caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]; "
             "print(caps)"],
            text=True, timeout=20,
        ).strip().splitlines()
        torch_ver  = result[0]
        n_gpus     = int(result[1])
        caps_str   = result[2]

        if n_gpus == 0:
            err(f"PyTorch {torch_ver} — no CUDA GPUs visible")
            all_ok = False
        else:
            ok(f"PyTorch [green]{torch_ver}[/]  [green]{n_gpus} GPU(s)[/]  {caps_str}")
            if "(12, 0)" in caps_str:
                info("RTX 5080 (Blackwell sm_120) detected ✓")
    except Exception as e:
        err(f"Cannot import torch: {e}")
        all_ok = False

    # required libs
    for lib, label, pip_name in [
        ("sentence_transformers", "sentence-transformers", "sentence-transformers"),
        ("easyocr",               "EasyOCR",               "easyocr"),
        ("fitz",                  "PyMuPDF (fitz)",         "pymupdf"),
        ("sklearn",               "scikit-learn",           "scikit-learn"),
        ("joblib",                "joblib",                 "joblib"),
        ("rich",                  "rich",                   "rich"),
        ("matplotlib",            "matplotlib",             "matplotlib"),
    ]:
        try:
            subprocess.check_output(
                [PYTHON, "-c", f"import {lib}"], stderr=subprocess.DEVNULL, timeout=10)
            ok(label)
        except Exception:
            err(f"{label} missing — run: [cyan]pip install {pip_name}[/]")
            if lib not in ("matplotlib",):
                all_ok = False
            else:
                warn(f"matplotlib optional (needed for hypersphere visualizations)")

    # model caches
    hf_cache  = Path.home() / ".cache/huggingface"
    ocr_cache = Path.home() / ".EasyOCR"
    if any(hf_cache.rglob("bge-m3*")):
        ok("bge-m3 cached (~2.2 GB — no download needed)")
    else:
        warn("bge-m3 not cached — first run downloads ~2.2 GB")
    if ocr_cache.exists():
        ok("EasyOCR weights cached")
    else:
        warn("EasyOCR not cached — first run downloads ~100 MB")

    # all GPUs
    gpus = all_gpu_stats()
    if gpus:
        console.print()
        t = Table(box=box.SIMPLE, show_header=True, border_style="dim")
        t.add_column("GPU", style="cyan", width=4)
        t.add_column("Name")
        t.add_column("Free VRAM", justify="right", style="green")
        t.add_column("Util", justify="right")
        t.add_column("Temp", justify="right", style="yellow")
        for gid, name, util, free, total, temp in gpus:
            note = " [green]← auto-selected[/]" if free >= 2500 else " [yellow](busy)[/]"
            t.add_row(str(gid), name,
                      f"{free:,}/{total:,} MiB{note}",
                      f"{util}%", f"{temp}°C")
        console.print(t)
        info("GPU with most free VRAM is chosen automatically at runtime")
    else:
        warn("nvidia-smi unavailable — cannot check GPU state")

    if not all_ok:
        console.print()
        console.print(Panel(
            "[red]One or more critical checks failed.[/]\n"
            "Fix the issues above, then rerun the wizard.",
            border_style="red",
        ))
    return all_ok


# ── Step 2: Mode selection ─────────────────────────────────────────────────────

def choose_mode() -> str:
    section("Step 2 — Choose Mode")
    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column("key", style="bold cyan", width=4)
    t.add_column("action")
    t.add_column("notes", style="dim")
    t.add_row("1", "Train a new model from a labelled corpus",
              "GPU auto-selected · rich TUI progress · ~minutes to hours")
    t.add_row("2", "Classify documents with an existing model",
              "single file or folder · top-3 classes with confidence")
    t.add_row("3", "Bulk sort a large collection (sort_docs.py)",
              "classify thousands of files · copy/symlink/report · CSV output")
    t.add_row("4", "Inspect an existing model",
              "show classes, weight norms, metadata — no GPU needed")
    t.add_row("Q", "Quit", "")
    console.print(t)

    while True:
        choice = Prompt.ask("Enter choice", choices=["1","2","3","4","q","Q"],
                            default="1")
        return choice.upper()


# ── Corpus inspection ──────────────────────────────────────────────────────────

def inspect_corpus(corpus: Path) -> bool:
    section("Corpus Inspection")
    class_dirs = sorted(p for p in corpus.iterdir() if p.is_dir())
    if not class_dirs:
        err(f"No sub-folders in [cyan]{corpus}[/]")
        return False

    console.print(f"  [green]{len(class_dirs)}[/] classes in [cyan]{corpus.name}[/]:\n")
    t = Table(box=box.SIMPLE, show_header=True, border_style="blue")
    t.add_column("Class", style="cyan", no_wrap=True)
    t.add_column("Files", justify="right", style="green")
    t.add_column("Formats", style="dim")

    total_files, small = 0, []
    for d in class_dirs:
        files = [f for f in d.rglob("*") if f.is_file() and not f.name.startswith("._")]
        exts  = sorted({f.suffix.lower() for f in files})[:4]
        n     = len(files)
        total_files += n
        prefix = "[yellow]" if n < 15 else ""
        t.add_row(f"{prefix}{d.name}", f"{prefix}{n}", " ".join(exts)[:40])
        if n < 15:
            small.append((d.name, n))

    t.add_row("[bold]TOTAL[/]", f"[bold green]{total_files:,}[/]", "")
    console.print(t)

    if small:
        console.print()
        warn(f"{len(small)} class(es) have < 15 documents (CV accuracy unreliable):")
        for name, n in small:
            info(f"  [cyan]{name}[/]: {n} docs")
    return True


# ── Training ───────────────────────────────────────────────────────────────────

def configure_training(corpus: Path) -> dict:
    section("Training Configuration")

    console.print("  [bold]Embedding model[/]")
    for k, (m, desc) in EMBED_MODELS.items():
        console.print(f"    [{k}] [cyan]{m}[/]  [dim]{desc}[/]")
    em_choice = Prompt.ask("    Select", choices=list(EMBED_MODELS.keys()), default="1")
    embed_model, _ = EMBED_MODELS[em_choice]

    console.print()
    console.print("  [bold]Chunk size[/] [dim](chars per chunk before mean-pooling)[/]")
    info("4000 works well for long documents; 1000-2000 for short ones.")
    chunk_chars = IntPrompt.ask("    --chunk-chars", default=4000)

    default_workers = min(16, max(1, (os.cpu_count() or 4) - 2))
    console.print()
    console.print(f"  [bold]CPU workers[/] [dim](parallel extraction processes, {os.cpu_count()} logical CPUs)[/]")
    workers = IntPrompt.ask("    --workers", default=default_workers)

    console.print()
    model_path = Prompt.ask("  [bold]Output model file[/]", default=str(DEFAULT_MODEL))

    console.print()
    info("[green]GPU selection is automatic[/] — the script picks the GPU with most free VRAM.")

    return {
        "embed_model": embed_model,
        "chunk_chars": chunk_chars,
        "workers":     workers,
        "model_path":  Path(model_path),
        "corpus":      corpus,
    }


def launch_training(cfg: dict) -> bool:
    section("Launch Training")

    cmd = (
        f"{PYTHON} {CLASSIFIER} train "
        f'"{cfg["corpus"]}" '
        f'-m "{cfg["model_path"]}" '
        f'--embed-model {cfg["embed_model"]} '
        f'--chunk-chars {cfg["chunk_chars"]} '
        f'--workers {cfg["workers"]}'
    )

    console.print("  Command:\n")
    console.print(Panel(f"[green]{cmd}[/]", border_style="green", padding=(0, 2)))
    console.print()
    info("Training opens in an alacritty window (mouse-select to copy errors).")
    info("• Rich TUI shows per-class progress bars + GPU stats after each class")
    info("• GPU selected automatically — whichever has the most free VRAM")
    info("• bge-m3 downloads ~2.2 GB on first run (one-time)")
    console.print()

    if not Confirm.ask("  Launch training now?", default=True):
        console.print("  [dim]Cancelled.[/]")
        return False

    alacritty_cmd = (
        f'DISPLAY=:0 alacritty --title "doc_classifier_gpu — training" '
        f'-e bash -c \'source {VENV_DIR}/bin/activate; {cmd}; '
        f'echo; echo "--- done (exit $?) --- press Enter to close ---"; read\' &'
    )
    os.system(alacritty_cmd)
    console.print()
    ok("Training launched in alacritty.")
    console.print("  [dim]Press Enter here when training completes ...[/]")
    input()
    return True


# ── Model inspection ───────────────────────────────────────────────────────────

def inspect_model(model_path: Path):
    section("Model Inspection")
    if not model_path.exists():
        err(f"Not found: [cyan]{model_path}[/]")
        info("Training may still be running — check the alacritty window for errors.")
        return

    size_kb = model_path.stat().st_size // 1024
    mtime   = time.strftime("%Y-%m-%d %H:%M", time.localtime(model_path.stat().st_mtime))
    ok(f"[cyan]{model_path.name}[/]  [green]{size_kb} KB[/]  saved {mtime}")

    try:
        import joblib as jl, numpy as np
        bundle  = jl.load(model_path)
        clf     = bundle["clf"]
        classes = list(clf.classes_)
        norms   = np.linalg.norm(clf.coef_, axis=1)

        console.print()
        console.print(f"  embed model : [cyan]{bundle['embed_model']}[/]")
        console.print(f"  chunk chars : [cyan]{bundle['chunk_chars']}[/]")
        console.print(f"  classes     : [green]{len(classes)}[/]")
        console.print(f"  weight matrix: [dim]({len(classes)} × {clf.coef_.shape[1]})[/]")
        console.print()

        t = Table(title="[bold green]Classes by distinctiveness[/]",
                  box=box.SIMPLE, border_style="green")
        t.add_column("Class", style="cyan")
        t.add_column("Weight norm", justify="right", style="dim")
        t.add_column("", style="dim")
        max_norm = norms.max()
        for cls, norm in sorted(zip(classes, norms), key=lambda x: -x[1]):
            bar = "█" * int(20 * norm / max_norm)
            t.add_row(cls, f"{norm:.3f}", bar)
        console.print(t)
        info("Higher weight norm = more distinctive / easier to separate from other classes.")

    except Exception as e:
        warn(f"Could not load model: {e}")


# ── Inference ──────────────────────────────────────────────────────────────────

def infer_wizard(model_path: Path):
    section("Classify Documents")

    if not model_path.exists():
        model_path = Path(Prompt.ask(
            "  Model file not found. Enter path",
            default=str(DEFAULT_MODEL),
        ))
        if not model_path.exists():
            err(f"Still not found: {model_path}")
            return

    while True:
        console.print()
        target_str = Prompt.ask(
            "  [bold]File or folder to classify[/]\n"
            "  [dim](PDF, image, text, or folder — GPU selected automatically)[/]"
        )
        target = Path(target_str.strip())
        if not target.exists():
            err(f"Not found: [cyan]{target}[/]")
            continue

        console.print()
        info("Running predict — GPU selected automatically ...")
        console.print()

        subprocess.run([PYTHON, str(CLASSIFIER), "predict",
                        str(target), "-m", str(model_path)])

        console.print()
        if not Confirm.ask("  Classify another?", default=False):
            break


# ── Bulk sort ──────────────────────────────────────────────────────────────────

def sort_wizard():
    section("Bulk Sort — sort_docs.py")

    if not SORTER.exists():
        err(f"sort_docs.py not found at [cyan]{SORTER}[/]")
        return

    # source
    console.print()
    source_str = Prompt.ask("  [bold]Source directory[/] (folder of files to classify)")
    source = Path(source_str.strip())
    if not source.is_dir():
        err(f"Not a directory: {source}")
        return

    file_count = sum(1 for f in source.rglob("*")
                     if f.is_file() and not f.name.startswith("._"))
    ok(f"[cyan]{source.name}[/]: [green]{file_count:,}[/] files")

    # model
    console.print()
    model_str = Prompt.ask("  [bold]Model file[/]", default=str(DEFAULT_MODEL))
    model_path = Path(model_str)
    if not model_path.exists():
        err(f"Model not found: {model_path}")
        return

    # output
    default_out = SCRIPT_DIR / (source.name + "_sorted")
    output_str  = Prompt.ask("  [bold]Output directory[/]", default=str(default_out))
    output = Path(output_str)

    # mode
    console.print()
    console.print("  [bold]Mode[/]")
    console.print("    [1] [cyan]report[/]   — classify only, write CSV, no file operations (preview)")
    console.print("    [2] [cyan]copy[/]     — copy files into class sub-folders (safe, uses disk space)")
    console.print("    [3] [cyan]symlink[/]  — create symlinks (space-efficient; source must stay mounted)")
    mode_choice = Prompt.ask("    Select", choices=["1","2","3"], default="1")
    mode = {"1": "report", "2": "copy", "3": "symlink"}[mode_choice]

    # threshold
    console.print()
    console.print("  [bold]Confidence threshold[/]")
    info("Files below threshold → _review/ folder. Higher = more files in _review/.")
    info("0.40 = sort most files   0.60 = sort high-confidence only   0.80 = very strict")
    threshold = Prompt.ask("    --threshold", default="0.60")

    # workers / batch
    default_workers = min(12, max(1, (os.cpu_count() or 4) - 2))
    workers = IntPrompt.ask("  [bold]--workers[/] (CPU extraction)", default=default_workers)
    batch   = IntPrompt.ask("  [bold]--batch[/] (docs per embedding window)", default=512)

    # skip-ocr
    console.print()
    info("--skip-ocr: image-only PDFs → _review/ instead of GPU OCR. ~10× faster for bulk runs.")
    skip_ocr = Confirm.ask("  [bold]--skip-ocr[/] (recommended for large collections)", default=True)

    # encode-batch
    console.print()
    info("--encode-batch: GPU forward-pass chunk batch size. bge-m3 on 16 GB GPU: keep ≤ 64.")
    encode_batch = IntPrompt.ask("  [bold]--encode-batch[/]", default=32)

    # dual-GPU
    console.print()
    try:
        import torch as _torch
        _n_gpus = _torch.cuda.device_count()
    except Exception:
        _n_gpus = 0
    dual_gpu = False
    if _n_gpus >= 2:
        info(f"{_n_gpus} GPUs detected. --no-single-gpu runs both simultaneously via subprocess")
        info("isolation (~2× throughput). bge-m3 must fit on each GPU (~5 GB at encode-batch=32).")
        dual_gpu = Confirm.ask("  [bold]--no-single-gpu[/] (dual-GPU subprocess mode)", default=True)
    else:
        info(f"Only {_n_gpus} GPU(s) detected — single-GPU mode.")

    # build command
    cmd = (
        f"{PYTHON} {SORTER} \"{source}\" "
        f"-m \"{model_path}\" "
        f"-o \"{output}\" "
        f"--mode {mode} "
        f"--threshold {threshold} "
        f"--workers {workers} "
        f"--batch {batch} "
        f"--encode-batch {encode_batch}"
    )
    if skip_ocr:
        cmd += " --skip-ocr"
    if dual_gpu:
        cmd += " --no-single-gpu"

    console.print()
    console.print("  Command:\n")
    console.print(Panel(f"[green]{cmd}[/]", border_style="green", padding=(0, 2)))
    console.print()

    gpu_factor = 2 if dual_gpu else 1
    embed_min = file_count // (1500 * gpu_factor) + 1
    info(f"Estimated time for {file_count:,} files:")
    info(f"  Extraction: ~{file_count // (workers * 60) + 1} min ({workers} workers)")
    info(f"  Embedding:  ~{embed_min} min (bge-m3 on {gpu_factor} GPU{'s' if gpu_factor > 1 else ''})")
    info(f"  Output:     {output}")
    console.print()

    if not Confirm.ask("  Launch sort run now?", default=True):
        console.print("  [dim]Cancelled.[/]")
        return

    alacritty_cmd = (
        f'DISPLAY=:0 alacritty --title "sort_docs — {source.name}" '
        f'-e bash -c \'source {VENV_DIR}/bin/activate; {cmd}; '
        f'echo; echo "--- done (exit $?) --- press Enter to close ---"; read\' &'
    )
    os.system(alacritty_cmd)
    ok("Sort run launched in alacritty (mouse-select text to copy errors).")

    if mode == "report":
        info(f"Results will be in [cyan]{output}/sort_report.csv[/]")
        info("Review the CSV, then rerun with --mode copy or symlink to actually sort files.")


# ── Inspect-only shortcut ──────────────────────────────────────────────────────

def inspect_wizard():
    section("Inspect Existing Model")
    model_str = Prompt.ask("  Path to model file", default=str(DEFAULT_MODEL))
    inspect_model(Path(model_str))


# ── Main ───────────────────────────────────────────────────────────────────────

def train_wizard():
    section("Step 3a — Corpus Selection")
    while True:
        corpus_str = Prompt.ask(
            "  [bold]Training corpus folder[/]\n"
            "  [dim](each immediate sub-folder = one class label)[/]",
            default=str(DEFAULT_CORPUS),
        )
        corpus = Path(corpus_str.strip())
        if not corpus.is_dir():
            err(f"Not a directory: [cyan]{corpus}[/]")
            continue
        if inspect_corpus(corpus):
            console.print()
            if Confirm.ask("  Use this corpus?", default=True):
                break

    cfg = configure_training(corpus)
    cfg["corpus"] = corpus
    launch_training(cfg)
    inspect_model(cfg["model_path"])

    if cfg["model_path"].exists():
        console.print()
        if Confirm.ask("  Run inference now?", default=True):
            infer_wizard(cfg["model_path"])


def main():
    banner()

    if not check_env():
        sys.exit(1)

    while True:
        mode = choose_mode()
        if mode == "Q":
            console.print("\n[dim]Goodbye.[/]\n")
            break
        elif mode == "1":
            train_wizard()
        elif mode == "2":
            model_str = Prompt.ask("\n  Path to model file", default=str(DEFAULT_MODEL))
            infer_wizard(Path(model_str))
        elif mode == "3":
            sort_wizard()
        elif mode == "4":
            inspect_wizard()

        console.print()
        if not Confirm.ask("Return to main menu?", default=True):
            console.print("\n[dim]Goodbye.[/]\n")
            break


if __name__ == "__main__":
    main()

# How-To Guide — doc_classifier_gpu

Operational reference for training, classifying, and sorting document collections
using the GPU embedding pipeline.

---

## 0. Prerequisites

```bash
# Activate the venv (required for every operation)
source ~/doc-clf-gpu-env/bin/activate
cd ~/Documents/claude_creations/2026-05-30_092836_doc-classifier-gpu/
```

Verify GPU availability:
```bash
python -c "import torch; print(torch.cuda.device_count(), 'GPU(s)')"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
```

---

## 1. Train a new model

### Prepare your corpus

Each immediate sub-folder of the training root becomes a class label:
```
training_root/
├── strategy/       ← class "strategy"
├── medical/        ← class "medical"
└── weapons/        ← class "weapons"
```

Minimum ~15 documents per class for reliable CV accuracy. Supported formats:
`.pdf`, `.jpg`, `.png`, `.webp`, `.txt`, `.md`, `.docx`, `.rtf`, `.eml`, `.xml`, `.html`

### Launch training (in alacritty — supports clipboard copy)

```bash
DISPLAY=:0 alacritty --title "doc_classifier_gpu — training" -e bash -c '
source ~/doc-clf-gpu-env/bin/activate
cd ~/Documents/claude_creations/2026-05-30_092836_doc-classifier-gpu
python doc_classifier_gpu.py train /path/to/corpus -m my_model.joblib
echo; echo "--- done (exit $?) --- press Enter ---"; read
' &
```

Mouse-select any text in alacritty to copy it to clipboard. Use xfce4-terminal or
gnome-terminal as alternatives. Avoid xterm — it has no clipboard copy support.

The script automatically selects the GPU with the most free VRAM. No need to set
`CUDA_VISIBLE_DEVICES` manually — `_pick_device()` handles it.

### Key training flags

| Flag | Default | Notes |
|------|---------|-------|
| `--embed-model` | `BAAI/bge-m3` | Lighter: `BAAI/bge-small-en-v1.5` |
| `--chunk-chars` | `4000` | Lower for short docs |
| `--workers` | `cpu_count-2` | Parallel CPU extraction processes |

### Read the output

- **CV accuracy** at the end is your honest accuracy estimate (held-out folds)
- A wide confidence spread (e.g. 0.88 ± 0.04) means the model is reliable
- Classes with < 15 docs are flagged — add more documents before trusting them

---

## 2. Use the interactive wizard

The wizard walks you through environment check → training → inference step by step:

```bash
python wizard.py
```

Modes: Train / Classify / Inspect / Quit. All steps explained inline.

---

## 3. Classify a single document

```bash
python doc_classifier_gpu.py predict /path/to/document.pdf -m my_model.joblib
```

Output: table showing top-3 predicted classes with confidence percentages.

---

## 4. Classify a folder of documents

```bash
python doc_classifier_gpu.py predict /path/to/folder/ -m my_model.joblib
```

Processes all files recursively. Results are printed as a Table, one row per file.
Pipe to a file for large collections:

```bash
python doc_classifier_gpu.py predict /path/to/folder/ -m my_model.joblib > results.txt
```

---

## 5. Bulk-sort a large collection (sort_docs.py)

For large collections (thousands of files), use `sort_docs.py` which provides
progress bars, GPU stats, and a CSV report.

### Step 1 — Report-only preview (no file operations)

```bash
python sort_docs.py /path/to/source/ \
    --model militia.joblib \
    --output ./sorted \
    --mode report \
    --threshold 0.60
```

Review `sorted/sort_report.csv` to see the classification distribution. Files below
60% confidence go to `_review/`.

### Step 2 — Copy files into class folders

```bash
python sort_docs.py /path/to/source/ \
    --model militia.joblib \
    --output ./sorted \
    --mode copy \
    --threshold 0.60
```

Creates `sorted/<class>/filename.pdf` for each classified file.

### Step 3 — Symlink instead of copy (saves disk space)

```bash
python sort_docs.py /path/to/source/ \
    --model militia.joblib \
    --output ./sorted \
    --mode symlink \
    --threshold 0.60
```

Source files stay in place; `sorted/` contains symlinks. Requires the source
volume to remain mounted.

### Recommended flags for large collections (thousands of files)

```bash
python sort_docs.py /path/to/source/ \
    --model militia.joblib \
    --output ./sorted \
    --mode report \
    --threshold 0.40 \
    --skip-ocr \
    --encode-batch 32 \
    --no-single-gpu
```

`--skip-ocr` skips GPU OCR (sends image-only PDFs to `_review/`) and runs ~10× faster.
`--no-single-gpu` activates dual-GPU subprocess mode — both RTX 5080 + RTX 3080 embed
simultaneously via process isolation, ~2× throughput with no heap corruption risk.

### sort_docs.py flags

| Flag | Default | Notes |
|------|---------|-------|
| `--mode` | `copy` | `copy` \| `symlink` \| `report` |
| `--threshold` | `0.40` | Files below this confidence → `_review/` |
| `--workers` | `cpu_count-2` | Parallel CPU extraction processes |
| `--maxtasks` | `50` | Pool maxtasksperchild; `1` = max crash isolation |
| `--batch` | `512` | Documents per GPU embedding window |
| `--encode-batch` | `32` | GPU forward-pass chunk batch; keep ≤64 for bge-m3 on 16 GB |
| `--skip-ocr` | off | Image-only files → `_review/`; ~10× faster for bulk runs |
| `--no-single-gpu` | off | Dual-GPU subprocess mode — both GPUs embed in parallel |

---

## 6. Inspect an existing model

```bash
python -c "
import joblib, numpy as np
b = joblib.load('militia.joblib')
clf = b['clf']
classes = clf.classes_
norms = np.linalg.norm(clf.coef_, axis=1)
print(f'{len(classes)} classes  embed={b[\"embed_model\"]}  chunk={b[\"chunk_chars\"]}')
print()
for cls, n in sorted(zip(classes, norms), key=lambda x: -x[1])[:10]:
    print(f'  {cls:<35} norm={n:.3f}')
"
```

Higher weight norm = more distinctive class in embedding space.

---

## 7. Two GPUs simultaneously

### Option A — Different workloads on each GPU (auto)

`doc_classifier_gpu.py` and `sort_docs.py` both call `_pick_device()` which probes
free VRAM and picks the GPU with the most headroom. If training occupies GPU 0,
the sort run automatically lands on GPU 1. No `CUDA_VISIBLE_DEVICES` needed.

```bash
# Terminal 1: training (auto-selects GPU 0)
python doc_classifier_gpu.py train /corpus -m model.joblib

# Terminal 2: sort run (auto-selects GPU 1 when GPU 0 busy)
python sort_docs.py /source -m model.joblib --output ./sorted --mode report --skip-ocr
```

### Option B — Both GPUs for the same sort run (`--no-single-gpu`)

```bash
python sort_docs.py /path/to/source/ -m militia.joblib \
  --output ./sorted --mode report \
  --skip-ocr --encode-batch 32 --no-single-gpu
```

This spawns two independent Python subprocesses via `mp.get_context('spawn')` —
one on `cuda:0` (RTX 5080), one on `cuda:1` (RTX 3080). Each loads bge-m3 in its
own heap (preventing the glibc corruption that occurred when loading twice in one
process). Batches of documents are round-robined between the two subprocesses.

**Verified result:** RTX 5080 at 88%, RTX 3080 at 100%; ~5,200 MiB each (vs ~13,400 MiB
single-GPU). Throughput is approximately 2× compared to single-GPU mode.

**Note:** Both bge-m3 instances must fit in VRAM simultaneously. RTX 3080 has 10 GB —
confirmed sufficient at `--encode-batch 32`.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `torch.OutOfMemoryError` at model load | GPU busy (training running) | `_pick_device()` auto-selects next GPU |
| `torch.OutOfMemoryError` in embed thread | `--encode-batch` too large | Use `--encode-batch 32`; OOM recovery halves it automatically |
| `BrokenProcessPool` in training | C extension segfault in worker (old pdfplumber) | Current code uses PyMuPDF — no Pillow exposure |
| `BrokenPipeError` cascade in sort_docs | `imap_unordered` + `WorkerLostError` kills pool | Restart loop already in current code; update if using old version |
| Heap corruption loading dual bge-m3 | Two `SentenceTransformer` loads in one process | Use `--no-single-gpu` (subprocess isolation via `spawn`) |
| Can't copy errors from terminal | xterm has no clipboard copy | Use alacritty (mouse-select → clipboard) |
| `(skipped)` for most files | Scanned PDFs, no text layer | Add `--max-ocr-pages 3` for sample OCR, or accept `_review/` |
| Many `worker error — retrying sequentially` | Old pdfplumber on image-heavy PDFs | Current code uses PyMuPDF — already fixed |
| `console.print` TypeError in except block | `stderr=True` is not a rich param | Already fixed |
| `._filename` files crashing EasyOCR | macOS resource forks in file list | Already filtered |
| CV accuracy not printed | All classes have < 2 docs (can't split) | Add more training examples |
| Low accuracy on a class | < 15 docs in that class | Add documents or merge with similar class |

---

## 9. File tree reference

```
2026-05-30_092836_doc-classifier-gpu/
├── doc_classifier_gpu.py           ← main classifier script
├── sort_docs.py                    ← bulk sort — streaming, dual-GPU, CSV report
├── wizard.py                       ← interactive training-to-inference wizard
├── militia.joblib                  ← trained model (64 classes)
├── militia_v8.joblib               ← trained model (v8 code)
├── README.md                       ← quick-start + architecture + incident notes
├── MILITIA_MODEL.md                ← model internals reference
├── LESSONS_LEARNED.md              ← 14 operational lessons
├── HOWTO.md                        ← this file
├── SESSION.md                      ← original session log (2026-05-30)
├── SESSION_2026-05-30_211956.md    ← resume session log
├── diagrams/                       ← architectural Graphviz diagrams
└── hypersphere/                    ← unit hypersphere concept docs + visualizations
    ├── HYPERSPHERE.md
    ├── hyper[1-5].png/svg          ← matplotlib geometric visualizations
    └── dot[1-5].png/svg            ← Graphviz architectural diagrams
```

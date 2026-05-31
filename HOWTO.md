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

### Launch training (in a visible xterm)

```bash
DISPLAY=:0 xterm -title "training" -fa 'Monospace' -fs 11 -geometry 130x55 -e bash -c '
source ~/doc-clf-gpu-env/bin/activate
cd ~/Documents/claude_creations/2026-05-30_092836_doc-classifier-gpu
python doc_classifier_gpu.py train /path/to/corpus -m my_model.joblib
echo; echo "--- done --- press Enter ---"; read
' &
```

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

### sort_docs.py flags

| Flag | Default | Notes |
|------|---------|-------|
| `--mode` | `copy` | `copy` \| `symlink` \| `report` |
| `--threshold` | `0.40` | Files below this confidence → `_review/` |
| `--workers` | `cpu_count-2` | Parallel CPU extraction workers |
| `--batch` | `256` | Documents per GPU embedding batch |

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

Training on GPU 0 + classification on GPU 1 works automatically:

```bash
# Terminal 1: training (auto-selects GPU 0 if free)
python doc_classifier_gpu.py train /corpus -m model.joblib

# Terminal 2: sort run (auto-selects GPU 1 when GPU 0 is busy)
python sort_docs.py /source -m model.joblib --output ./sorted --mode report
```

The `_pick_device()` function probes free VRAM on all GPUs and picks the best one.
No `CUDA_VISIBLE_DEVICES` needed.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `torch.OutOfMemoryError` at model load | GPU busy (training running) | Script now auto-selects next GPU |
| `BrokenProcessPool` | C extension segfault in worker (old pdfplumber code) | Use v10+ code (PyMuPDF) |
| `(skipped)` for most files | Scanned PDFs, no text layer | EasyOCR handles these via GPU OCR |
| Many `worker error — retrying sequentially` | Old pdfplumber code on image-heavy PDFs | Upgrade to v10 (PyMuPDF) |
| `console.print` TypeError in except block | `stderr=True` is not a rich param | Already fixed in v9 |
| `._filename` files crashing EasyOCR | macOS resource forks in file list | Already filtered in v9 |
| CV accuracy not printed | All classes have < 2 docs (can't split) | Add more training examples |
| Low accuracy on a class | < 15 docs in that class | Add documents or merge with similar class |

---

## 9. File tree reference

```
2026-05-30_092836_doc-classifier-gpu/
├── doc_classifier_gpu.py   ← main script (v10)
├── sort_docs.py            ← bulk classifier sorter
├── wizard.py               ← interactive wizard
├── militia.joblib          ← trained model (64 classes, run 3)
├── militia_v8.joblib       ← trained model (v8 code, when complete)
├── README.md               ← quick-start + badges + diagrams
├── MILITIA_MODEL.md        ← model internals reference
├── HOWTO.md                ← this file
├── SESSION.md              ← session log, incidents, lessons learned
├── diagrams/               ← architectural Graphviz diagrams
└── hypersphere/            ← unit hypersphere concept docs + visualizations
    ├── HYPERSPHERE.md
    ├── hyper[1-5].png      ← matplotlib geometric visualizations
    └── dot[1-5].png        ← Graphviz architectural diagrams
```

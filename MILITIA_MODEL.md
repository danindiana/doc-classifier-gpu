# militia.joblib — Model Reference

**File:** `militia.joblib`  
**Produced by:** `doc_classifier_gpu.py train`  
**Used by:** `doc_classifier_gpu.py predict`

---

## What it is — plain language

`militia.joblib` is a trained document classifier. Give it any PDF, image, or text
file and it returns a ranked list of the most likely document categories with
confidence percentages.

Think of it like a librarian who has read thousands of already-sorted documents and
learned the signature of each category — the vocabulary, topics, writing style, level
of technicality. When you hand it an unsorted document, it compares the document's
"fingerprint" to everything it has seen and tells you which shelf it belongs on, and
how confident it is.

The classifier has 69 categories, one per labelled sub-folder in the training corpus.
It was trained on the `2026-05-28_militia-copy/` collection.

---

## What it is — technical

A Python dict serialized with `joblib` (pickle + LZ4 compression):

```python
{
    "clf":         LogisticRegression(C=10.0, class_weight="balanced", max_iter=5000),
    "embed_model": "BAAI/bge-m3",
    "chunk_chars": 4000,
}
```

### `clf` — the classifier

A fitted `sklearn.linear_model.LogisticRegression`. Its weight matrix has shape
`(n_classes, 1024)` — one 1024-dimensional weight vector per class. At inference:

1. The query document is embedded to a 1024-dim unit vector (see below)
2. A dot product against each class weight vector gives a raw score per class
3. Softmax converts those scores to a probability distribution over all 64 classes
4. The top-N classes and their probabilities are returned

LogisticRegression is well-calibrated — the output percentages approximate real-world
accuracy at that confidence level. It is also interpretable: `clf.coef_[i]` is the
direction in embedding space that most strongly predicts class `i`.

Key hyperparameters:
- `C=10.0` — relatively low regularization (suited for high-quality embeddings in
  a low-noise regime)
- `class_weight="balanced"` — weights loss by inverse class frequency, preventing
  large classes from dominating (important given classes range from ~5 to 706 docs)
- `max_iter=5000` — enough iterations for convergence on 1024-dim inputs

### `embed_model` — the encoder

`BAAI/bge-m3` — a 570M-parameter multilingual sentence encoder from BAAI.
Outputs 1024-dimensional dense vectors. Context window: 8192 tokens.
No prompt prefix required (unlike some other bge variants).

The model name is stored — not the weights. At predict time, the same encoder is
reloaded from `~/.cache/huggingface/`. If the cache is absent, it downloads (~2.2 GB).

### `chunk_chars` — text segmentation

Long documents are split into overlapping 4000-character chunks (200-char overlap) before
encoding, then the chunk embeddings are mean-pooled into a single document vector.
Baked in so train and predict always use the same segmentation.

---

## How the embedding pipeline works

For each document:

```
raw file
   │
   ├─ PyMuPDF (fitz)         PDF with text layer  → plain text
   ├─ EasyOCR + PyMuPDF      scanned PDF / image  → OCR text
   └─ read_text()            .txt / .md / .eml    → plain text
   │
   ▼
chunk_text(text, chunk_chars=4000, overlap=200)
   → ["chunk_0", "chunk_1", ..., "chunk_N"]
   │
   ▼
bge-m3.encode(chunks, batch_size=32, normalize_embeddings=True)
   → float32 array  shape (N_chunks, 1024)    ← L2-normalized
   │
   ▼
mean over chunks → shape (1024,)              ← document vector
   │
   ▼
LogisticRegression.predict_proba([doc_vec])   ← softmax over 64 classes
   → [(class_name, probability), ...]
```

The L2-normalization places all vectors on the unit hypersphere. LogisticRegression in
this space approximates a cosine-similarity classifier — semantically similar documents
cluster together regardless of vocabulary.

---

## The 64 classes

One class per training sub-folder. The full list from the training corpus:

```
3d_bat                          3dprintedaction                 3rd_Bat_ARSMC
advisors                        AR helmet                       Ark Fed Forces
armor                           Artificial Intelligence         Artillery
C2                              Camoflaug an signature management
camps                           chemical_weapons                civil affairs
COIN                            Combat Medicine                 Communications and Radio INFO
Computers_Data                  deception operations            DRONES
energetics                      FEMA                            Forest_hiking
For_Internal_defense            fussboll                        Games and simulations
geography                       gis                             hybrid threat
ideology                        improvised weapons              Information Processing
INFOWAR                         Intel                           intelligence
Irregulate_Warfare              jungle operations               legal
Liberated_manuals               logistics                       LSCO
Media Affairs                   medical                         Militia_quickstart library
mines                           mortars                         motor
NBC                             northcom                        org_chem
Patches and Insignia            patents                          physical training
physiology                      Psyop                           radio
rADIO_baofeng_chirp             Remote_Sensing                  Robotics
Russia_Countering               Spec_ops                        stability operations
strategy                        System of Systems Design        tactics
training                        unconventional_warfare           weapons
```

The exact list is available at runtime via `bundle["clf"].classes_`.

---

## How to use it

```bash
source ~/doc-clf-gpu-env/bin/activate
cd ~/Documents/claude_creations/2026-05-30_092836_doc-classifier-gpu/

# classify a single file
CUDA_VISIBLE_DEVICES=0 python doc_classifier_gpu.py predict \
    /path/to/unknown_doc.pdf -m militia.joblib

# classify every file in a folder (recursive)
CUDA_VISIBLE_DEVICES=0 python doc_classifier_gpu.py predict \
    /path/to/unlabelled_folder/ -m militia.joblib
```

Output is a formatted table — one row per file:

```
File                                     1st               2nd             3rd
────────────────────────────────────     ───────────────   ─────────────   ─────────
fieldmanual_007.pdf                      strategy 88%      C2 7%           Spec_ops 5%
patrol_photo.jpg                         camps 61%         Forest_hiking 24%  geography 15%
unknown_scan.pdf                         <no extractable text>
```

### Use it directly in Python

```python
import joblib
from pathlib import Path

bundle  = joblib.load("militia.joblib")
clf     = bundle["clf"]
classes = clf.classes_

# If you already have a document embedding (1024-dim numpy array):
proba   = clf.predict_proba([doc_embedding])[0]
top3    = sorted(zip(classes, proba), key=lambda x: -x[1])[:3]
for cls, p in top3:
    print(f"  {cls}: {p:.1%}")

# Inspect weight vectors (what each class "looks like" in embedding space):
import numpy as np
for i, cls in enumerate(classes):
    top_dims = np.argsort(clf.coef_[i])[::-1][:5]
    print(f"{cls}: top dims {top_dims}")  # which embedding dimensions drive this class
```

---

## Interpreting the output

| Confidence pattern | Meaning | Action |
|--------------------|---------|--------|
| 1st ≥ 80%, 2nd < 15% | Strong match — document clearly resembles this category | Trust it |
| 1st 50–79%, 2nd 20–40% | Moderate match — document spans two categories | Spot-check |
| Tight spread (all < 40%) | Ambiguous — document bridges multiple categories, or topic not well-represented in training | Manual review |
| `<no extractable text>` | No text layer found, OCR returned nothing | Check file integrity; may be a photo, diagram, or corrupt scan |

**Calibration note:** Logistic Regression outputs are better calibrated than random
forests or SVMs. The stated 88% is a real probability estimate, not just a ranking.

---

## What the model has learned

The model did **not** learn keywords. It learned directions in a 1024-dimensional
semantic space. This means:

- Two documents using completely different vocabulary but covering the same topic
  will score the same class.
- A document in a different language will still classify correctly if bge-m3
  maps it to the same semantic region (bge-m3 is multilingual).
- Paraphrases, reformulations, and summaries of training documents will score high
  confidence for their class.

What it has **not** learned: layout, font, document structure, metadata. It sees
only the extracted text.

---

## Limitations

**Closed-world assumption:** The model only knows 69 categories. A genuinely novel
document type will be assigned the most similar existing category, not flagged as
"unknown." Watch for low max-confidence scores (<40%) as a signal of out-of-distribution
documents.

**Training set size:** Accuracy degrades for small classes. Roughly:
- < 15 docs/class → CV score is noisy; don't trust the number
- 15–50 docs/class → meaningful signal
- 50+ docs/class → reliable

Some classes in this corpus have very few documents. Their boundaries are approximate.

**Image quality:** EasyOCR accuracy drops on low-resolution scans, handwriting,
non-standard fonts, or heavily compressed images. Misclassification from bad OCR
is indistinguishable from a genuine category ambiguity.

**English-only OCR:** The EasyOCR reader is initialized with `['en']`. Foreign-language
image text will produce garbage or empty output. The embedding model (bge-m3) is
multilingual, but the OCR step is not.

**No confidence threshold:** The model always produces a prediction. If the highest
score is 12%, the model still names a class. Callers should check `proba.max()` and
reject or flag low-confidence outputs.

---

## Retraining and updating

### Add a new category

```bash
mkdir ~/Documents/claude_creations/2026-05-28_militia-copy/new_category/
# copy training documents into it
CUDA_VISIBLE_DEVICES=0 python doc_classifier_gpu.py train \
    ~/Documents/claude_creations/2026-05-28_militia-copy \
    -m militia.joblib   # overwrites previous model
```

### Add documents to an existing category

Copy new files into the existing sub-folder and rerun `train`. The entire model
retrains from scratch — there is no incremental update. Retraining is dominated by
the embedding step (~seconds per document on GPU).

### Change the embedding model

Pass `--embed-model BAAI/bge-small-en-v1.5` at train time. The new model name is
baked into the bundle; predict reloads it automatically. The two models are not
compatible — you cannot use a bge-small bundle with a bge-m3 encoder or vice versa.

---

## Portability

`militia.joblib` is ~500 KB. It contains only:
- The LR weight matrix (69 × 1024 float32 = ~280 KB)
- Class names (strings)
- `embed_model` string and `chunk_chars` int

It does **not** contain:
- bge-m3 weights (~2.2 GB, stored in `~/.cache/huggingface/`)
- EasyOCR weights (~100 MB, stored in `~/.EasyOCR/`)
- The training corpus

To deploy on another machine:

```bash
# Option A: internet available
scp militia.joblib target:~/
# bge-m3 + EasyOCR will download on first predict

# Option B: air-gapped
scp militia.joblib target:~/
rsync -a ~/.cache/huggingface/ target:~/.cache/huggingface/
rsync -a ~/.EasyOCR/ target:~/.EasyOCR/
```

The target machine must have the `doc-clf-gpu-env` Python environment with all deps
installed. GPU is optional for predict (falls back to CPU if CUDA unavailable, slow
for large batches).

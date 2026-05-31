# Screenshots

Live captures from the doc_classifier_gpu suite running on worlock (2026-05-30).

| File | Description |
|------|-------------|
| `01_dual_gpu_nvidia_smi.png` | `nvidia-smi` showing both GPUs: RTX 5080 (Blackwell sm_120, GPU 0) + RTX 3080 (GPU 1). Driver 580, CUDA 13.0. |
| `02_model_inspection.png` | `militia.joblib` class list sorted by weight norm (distinctiveness). 64 classes, bge-m3 embeddings, 64×1024 weight matrix. |
| `03_wizard_env_check.png` | `wizard.py` v2.0.0 startup — banner + Step 1 environment check showing PyTorch 2.12+cu130, 2 GPUs, Blackwell sm_120 detected. |
| `04_sort_docs_help.png` | `sort_docs.py --help` showing the full flag set including `--skip-ocr` and `--max-ocr-pages`. |

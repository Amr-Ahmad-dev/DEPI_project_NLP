# DEPI_project_NLP

[![Project](https://img.shields.io/badge/Project-Internship-blue)]()  [![Language](https://img.shields.io/badge/Language-Python-orange)]()

Welcome — this repository contains the complete project I built for my internship in NLP engineering. Below you’ll find what I built, how it works, the data I used, design decisions, current limitations, and suggested next steps.

---

## Project snapshot
- This repository is a complete internship project that explores automated text correction using neural networks and tokenization strategies.
- Two main code artifacts:
  - `NLB` — my original generation file (the initial implementation I wrote).
  - `n.py` — an AI-assisted, improved version that leverages PyTorch and more GPU utilization to speed up and (attempt to) improve the process.

---

## High-level description (points summarized)
1. This is a complete project for my internship — the full experiment, data, and notes are included.
2. `NLB` is the actual project I originally implemented and tested.
3. `n.py` is an improved version in which I used AI-assisted development and PyTorch to utilize GPU capability and attempt to accelerate and refine generation.
4. I built a neural network as the core model for the task.
5. I selected tokenization as a critical preprocessing step and implemented a tokenization pipeline.
6. I evaluated and chose what I judged the best tokenization and vectorization method for this dataset (details below).
7. `NLB` is a file I generated myself (the baseline generator).
8. `n.py` is the file where AI tried to improve my generation approach (optimizations, model adjustments, GPU usage).
9. All of the above are explained in the README sections below in a clean, visual way.
10. `grammar correction.csv` and `/all` CSVs are the datasets used: `grammar correction.csv` focuses on grammar errors; the `/all` CSVs are primarily targeted at spelling and word-level corrections.
11. The "Adaptive Accessibility Roadmap" PowerPoint (included in the repo) is my proposed roadmap for the project — it contains engineering plans, priorities, and suggested future directions. I also explain engineering decisions, which parts can change, and observed model limitations. We hypothesize many limitations stem from small model size and limited vocabulary in the dataset.

---

## Files & directories (what to look at)
- `NLB` — original generation script / file (my baseline).
- `n.py` — AI-assisted improved script using PyTorch and GPU optimizations.
- `data/grammar correction.csv` — dataset focused on grammar corrections.
- `data/all/*.csv` — datasets focused on word-level/spelling errors.
- `Adaptive Accessibility Roadmap.pptx` — the project roadmap and engineering plan.
- `requirements.txt` (if present) — Python dependencies.
- `README.md` — this file.

---

## Model & architecture
- I built a neural network tailored to text correction:
  - Architecture: a lightweight sequence model (custom neural net) designed to predict corrected tokens from noisy input.
  - Training: supervised training on CSV pairs (noisy -> corrected).
  - Loss and metrics: standard cross-entropy loss with token-level accuracy and word-level correction checks.
- Engineering emphasis:
  - Efficiency: `n.py` introduces PyTorch-based batching, mixed-precision training options, and GPU utilization to speed up experiments.
  - Reproducibility: model checkpoints and training seed control are included in the code (where applicable).

---

## Tokenization & vectorization
- Tokenization is a core design decision:
  - I evaluated tokenization methods (character-level, word-level, and subword/BPE-like approaches).
  - Final choice: the tokenization and vectorization approach that best matched our limited dataset vocabulary and correction targets (details are documented in comments inside `n.py` / `NLB`).
  - Rationale: smaller dataset vocabularies often benefit from subword tokenization to balance vocabulary size and generalization; for spelling-heavy errors, character or hybrid tokenization can help.
- Vectorization:
  - Tokens are mapped to embeddings (learned embeddings in the model).
  - Where useful, pretrained embeddings were considered but the primary experiments used task-specific embeddings due to domain-specific vocabulary.

---

## Data
- `grammar correction.csv`
  - Purpose: training examples centered on grammar fixes (syntax, agreement, punctuation).
  - Use-case: improve grammatical correctness.
- `/all` CSVs
  - Purpose: larger collection aimed mainly at spelling/word-level corrections.
  - Use-case: spelling normalization and word corrections.
- Notes on data:
  - CSV format: typically [input_text, corrected_text] per row.
  - Preprocessing: cleaning, lowercasing (configurable), tokenization before training.

---

## Engineering decisions, trade-offs, and what can change
- Decisions made:
  - Implemented a compact neural model (smaller parameter count) to allow experiments to run on limited compute.
  - Chose tokenization method that performed best on validation splits.
  - Used PyTorch in `n.py` to enable GPU acceleration and better experiment control.
- Trade-offs:
  - Smaller model = faster training but lower expressivity.
  - Less preprocessing keeps the pipeline simpler but could leave noise that limits performance.
- What can change:
  - Scale up model size (transformer layers or larger RNNs) for better capacity.
  - Increase dataset size and vocabulary coverage via data augmentation or external corpora.
  - Try pretrained language models (fine-tuning transformer-based models) for immediate gains.
  - Use subword tokenizers (Byte-Pair Encoding, SentencePiece) if not already in use.
  - Experiment with mixed precision, distributed training, and larger batch sizes.

---

## Observed limitations & hypothesis
- Limitation: model performance plateaued on some correction types.
- Hypothesis: the main causes are:
  - Limited model size (insufficient capacity for complex patterns).
  - Limited vocabulary and dataset coverage (not enough examples of many error types).
- Recommended next steps to address limitations:
  - Expand dataset (more examples, synthetic augmentation).
  - Increase model capacity or fine-tune pretrained models.
  - Improve tokenization (subword methods) and include contextual embeddings.

---

## How to run (example)
1. Create a Python environment:
   - python -m venv venv
   - source venv/bin/activate (or venv\Scripts\activate on Windows)
2. Install requirements:
   - pip install -r requirements.txt
   - If no requirements file, at minimum: pip install torch numpy pandas
3. Train / test:
   - Baseline (NLB): python NLB
   - AI-improved (n.py): python n.py --config configs/train.yaml
   - GPU note: ensure CUDA-enabled PyTorch is installed and GPU is available. `n.py` includes GPU checks and batching for better utilization.

---

## Roadmap (Adaptive Accessibility Roadmap)
- The included PowerPoint contains:
  - Proposed timeline and milestones.
  - Prioritization for model improvements and integration into accessible tools.
  - Engineering tasks (data expansion, model scaling, UI/UX integration).
- It reflects my recommended path for the project given current results and constraints.

---

## Closing notes
- This repository documents both a human-designed baseline (`NLB`) and an AI-assisted attempt to improve it (`n.py`).
- The work highlights practical trade-offs between experiment speed and model capacity, and points clearly to data and model scaling as the highest-impact next steps.

---

Credits & contact
- Author: Amr Ahmad
- GitHub: @Amr-Ahmad-dev

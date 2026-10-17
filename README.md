# 🧩 ARC Prize 2026 — Abstraction & Reasoning Corpus (ARC-AGI 2)

> **Kaggle Competition:** [ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)  
> **Compute Resource:** 💻 Pure CPU (0 GPU slots consumed — no concurrency restriction)  
> **Internet:** ❌ Disabled  
> **Metric:** Exact Match Top-2 (Top-2 Accuracy per problem)  
> **Deliverable:** `submissions/submission.json`  

---

## 🎯 Competition Objective

Solve visual abstract inductive-reasoning tasks where only a few demonstration pairs (input-grid $\to$ output-grid) are given and the target grid must be predicted for one or two test inputs.

Kaggle allows **two attempts per task** (`attempt_1` and `attempt_2`). If either one matches the expected matrix exactly, the task receives full credit.

---

## 🏗️ Technical Approach and Methodology

The pipeline implemented in [`kaggle_arc_baseline.ipynb`](kaggle_arc_baseline.ipynb) uses a **Test-Time Search & Program Synthesis (DSL)** approach:

1. **Geometric Primitives and Symmetries:**
   - D4 symmetry group (rotations by $90^\circ, 180^\circ, 270^\circ$, horizontal/vertical and diagonal reflections).
   - Background cropping / non-background crop (`crop_non_bg`).
2. **Cellular and Physical Transformations:**
   - Gravity in 4 directions (Up, Down, Left, Right).
   - Filling of enclosed regions (*fill-enclosed*).
   - Color-frequency remapping (*freq-rank remap*).
   - Selection of periodic sub-grids (*tiling & grid-select*).
3. **Confidence Engine and Compound Rules:**
   - Each hypothesis receives a consistency score over the task's training examples.
   - The top 2 hypotheses produce `attempt_1` and `attempt_2`.
4. **Time-Budgeting:**
   - Maximum of 10–15 seconds per task to make sure the kernel finishes within 9 hours on Kaggle's hidden test set.

---

## 📁 Directory Structure

```text
ARC Prize 2026/ARC Prize 2026/
├── kaggle_arc_baseline.ipynb   # Main notebook synced with Kaggle
├── kernel-metadata.json        # Remote configuration metadata
├── EXPERIMENT_LOG.md           # Historical log of versions and results
└── submissions/
    └── submission.json         # Downloaded artifact ready to evaluate
```

---

## 🚀 How to Run

From the repository root (`Kaggle/`):

```powershell
# Launch on Kaggle and download submission.json when finished
python kaggle_manager.py run arc

# Or directly via PowerShell
.\kaggle_remote.ps1 run arc
```

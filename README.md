# 🧩 ARC Prize 2026 — Abstraction & Reasoning Corpus (ARC-AGI-2)

> **Kaggle Competition:** [ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)  
> **Compute Resource:** 🎮 GPU 4×L4 (`machine_shape: NvidiaL4` — 1 GPU slot, 2× quota rate)  
> **Internet:** ❌ Disabled  
> **Metric:** Exact Match Top-2 (Top-2 Accuracy per problem, 0–100 scale)  
> **Best LB Score:** **`31.39`** (`v5_nvarc_qwen3_ttt`)  
> **Final Leaderboard Rank:** 🏅 **740 / 3543** teams (top 21%)  
> **Deliverable:** `submissions/submission.json`  

---

## 🎯 Competition Objective

Solve visual abstract inductive-reasoning tasks where only a few demonstration pairs (input-grid $\to$ output-grid) are given and the target grid must be predicted for one or two test inputs.

Kaggle allows **two attempts per task** (`attempt_1` and `attempt_2`). If either one matches the expected matrix exactly, the task receives full credit.

---

## 🏗️ Technical Approach and Methodology (v5–v6: NVARC Test-Time Fine-Tuning)

After four CPU DSL versions (`v1`–`v4`) scored `0.00` on the hidden test set, [`kaggle_arc_baseline.ipynb`](kaggle_arc_baseline.ipynb) transitioned in `v5`/`v6` to **NVARC (LLM + Test-Time Training on 4×L4 GPUs)**, achieving **`31.39` LB**:

1. **Base Model (`sorokin/qwen3_4b_grids15_sft139`):**
   - Offline Qwen3-4B model pre-trained on 2D ARC grid representations, mounted as a Kaggle Model input.
2. **Per-Task Test-Time Fine-Tuning (TTT):**
   - Trains a fresh LoRA adapter ($r=256$) per test task using $16\text{–}24$ D4 dihedral symmetries (rotations + reflections) combined with bijective color permutations of the task's demonstration pairs.
3. **Constrained Depth-First Search (DFS) Decoding:**
   - Generates candidate output grids restricted to the 12 valid ARC grid tokens under a cumulative log-probability threshold.
4. **Augmentation NLL Consensus Re-Ranking (`v6`):**
   - Evaluates candidate grids across 8–12 inverse-transformed augmentations using negative log-likelihood (NLL) + consensus frequency voting to select `attempt_1` and `attempt_2`.
5. **Multi-GPU Parallelism:**
   - Distributes tasks across 4 worker processes (one per L4 GPU, 96 GB total VRAM) within the 12-hour runtime budget.

> [!IMPORTANT]
> **Prize Eligibility (CC BY 4.0 / Open Source AI Definition):** To qualify for cash prizes, both the inference notebook and the full SFT/TTT training recipe must be publicly released under **CC BY 4.0** before the winner review.

---

## 📁 Directory Structure

```text
ARC Prize 2026 - ARC-AGI-2/ARC Prize 2026 - ARC-AGI-2/
├── kaggle_arc_baseline.ipynb   # Main NVARC 4xL4 notebook synced with Kaggle
├── update_arc.py               # Notebook builder/updater script for v6
├── kernel-metadata.json        # Remote configuration metadata (GPU L4x4, offline)
├── EXPERIMENT_LOG.md           # Historical log of versions and results
├── OVERVIEW.md                 # Official competition overview
├── DATASET_DESCRIPTION.md      # Dataset format specification
├── RULES.md                    # Prize eligibility & license rules
└── submissions/
    └── submission.json         # Downloaded artifact from commit run
```

---

## 🚀 How to Run

From the repository root (`Kaggle/`):

```powershell
# Launch commit smoke-run on Kaggle (4 eval tasks) and download submission.json
python kaggle_manager.py run arc

# Or directly via PowerShell
.\kaggle_remote.ps1 run arc
```

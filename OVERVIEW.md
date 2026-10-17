# 🧩 ARC Prize 2026 — Summary of the Official Kaggle Overview

**Competition:** [ARC Prize 2026 - ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)  
**Organizer:** François Chollet & ARC Prize Foundation  
**Focus:** Fluid Intelligence, Abstract Reasoning and Out-of-Distribution (OOD) Generalization

---

## 1. 🎯 General Description and Purpose
The **Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI-2)** is the gold-standard benchmark for measuring an AI system's ability to solve novel problems that cannot be solved by simple memorization, massive parameter scale or shallow interpolation.

Unlike traditional vision or NLP benchmarks, where models memorize statistical associations from large volumes of pretraining data, ARC evaluates **skill-acquisition efficiency**:
- Each task poses a visual and logical transformation that humans solve intuitively from **a few demonstration examples (few-shot demonstration pairs)**.
- The model must induce the latent rule and apply it to an unseen test grid.
- The tasks probe the so-called human *Core Knowledge Priors*: object permanence, topology, geometric symmetries ($D_4$), elementary arithmetic and spatial morphology.

---

## 2. 📏 Metric and Evaluation Rules
- **Metric:** **Exact Match Top-2 per test problem**.
- **Evaluation Mechanism:**
  - For each input grid in the test set (`test`), the solution produces up to **two attempts (`attempt_1` and `attempt_2`)**.
  - A score of **1** is awarded if at least one of the two attempts matches the true solution exactly in dimensions and pixel values. Otherwise the score is **0**.
  - The final Leaderboard metric is the average fraction of tasks solved successfully:
    $$\text{Score} = \frac{1}{N} \sum_{i=1}^N \max\left( \mathbb{I}(\hat{Y}_{i,1} = Y_i), \mathbb{I}(\hat{Y}_{i,2} = Y_i) \right)$$
- **Output Format:** A single `submission.json` file indexed by task identifier (`task_id`).

---

## 3. ⏱️ Kaggle Environment and Compute Constraints
- **Competition Type:** *Code Competition*. The notebook is re-run on 240 hidden tasks.
- **Internet Access:** ❌ Disabled (`"enable_internet": false`).
- **Time Limit:** CPU or GPU notebook **≤ 12 hours**. Scoring runtimes are obfuscated by up to ~10 minutes.
- **Accelerators:** CPU, standard GPUs, or **L4x4** (96 GB VRAM). L4x4 is restricted to notebooks attached to this competition, must run with internet off, and consumes GPU quota at 2×.
- **External data:** Free, publicly available data and pre-trained models are allowed.
- **Our kernel:** `rochikarla/arc-prize-2026-baseline` currently runs on an **L4** with the Qwen3-4B fine-tune `sorokin/qwen3_4b_grids15_sft139`.
- **Recommended Strategy:** Per-task time budgeting with fast fallbacks when the solver does not converge.

---

## 4. 🗓️ Timeline (23:59 UTC)
| Date | Milestone |
|---|---|
| 2026-03-25 | Start |
| **2026-10-26** | Entry + team merger deadline |
| **2026-11-02** | Final submission deadline |
| 2026-12-04 | Winners announced |

---

## 5. 🏆 Prizes ($700,000)
- **Progress Prizes ($275,000):** $75k / $50k / $40k / $35k / $25k / $20k / $15k / $15k.
- **Innovation / Grand Prize ($275,000):** the best **Solution Writeup**, scored 0–5 on Accuracy, Universality, Progress, Theory, Completeness and Novelty. Artifacts must be open-sourced and attached within 7 days of the submission deadline.
- **Bonus ($150,000):** unlocked only if a team reaches **≥ 85 %**, split among the top 5 such teams.
- Prize-eligible teams that **do not open-source** their solution are removed. See `RULES.md`.

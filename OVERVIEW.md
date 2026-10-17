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
- **Competition Type:** *Code Competition* (blind evaluation on a hidden test set).
- **Internet Access:** ❌ Strictly disabled (`"enable_internet": false`).
- **Acceleration:** CPU or standard GPU. (In our multi-agent pipeline it runs on CPU so as not to consume GPU quota).
- **Time Limit:** 9 or 12 total hours to process the full hidden test set.
- **Recommended Strategy:** Dynamic per-problem *time-budgeting* (5 to 15 seconds maximum per task) with fast fallback mechanisms when program synthesis does not converge.

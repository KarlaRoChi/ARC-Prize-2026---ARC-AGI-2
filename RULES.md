# ⚖️ ARC Prize 2026 — ARC-AGI-2: Rules & Prize Eligibility

**Official rules:** <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules>
**Last checked against the official pages:** 2026-10-17
*This is a summary for internal use. The official rules page is authoritative.*

---

## 1. Key terms

| Item | Value |
|---|---|
| Sponsor | ARC Prize Foundation |
| Prize pool | $700,000: Progress Prizes $275k (8 places, $75k → $15k), a writeup-judged Innovation/Grand Prize of $275k, and a Bonus of $150k unlocked only at ≥ 85 % |
| Winner license | **CC BY 4.0** |
| Data license | Apache 2.0 |
| Team size | Max 5 |
| Submissions | **1 per day**, 2 final selections |
| Code requirements | Notebook only. CPU or GPU ≤ **12 h**. **Internet disabled.** Free, public external data and pre-trained models allowed. Output `submission.json` |
| Accelerators | L4x4 (96 GB). Only for notebooks attached to this competition, internet must be off, quota counts at 2× |

## 2. Deadlines (23:59 UTC)

| Date | Milestone |
|---|---|
| **2026-10-26** | Entry + team merger deadline |
| **2026-11-02** | **Final submission deadline** |
| Submission deadline + 7 days | Open-sourced artifacts must be attached to a Solution Writeup to qualify for the Innovation/Grand Prize |
| 2026-12-04 | Winners announced |

## 3. What a submission needs to be prize-eligible

1. **Open source is mandatory.** Prize-eligible participants who do not open-source their solution are **removed from the competition**.
2. **Fully open stack.** Submissions need an open-source system, an open-source model **and open weights/parameters**, as defined by the OSI *Open Source AI Definition* checklist. That definition also expects the training code and enough information about the training data.
3. Release the winning submission and its code under **CC BY 4.0**.
4. Offline rerun within 12 h, with all models mounted from Kaggle.
5. No hand-labelling of test tasks and no private sharing.
6. **Winner obligations.** Deliver training and inference code with an environment description. Do an interview with the sponsor and work with their technical writer. Sign the prize documents.
7. **Innovation/Grand Prize.** Scored 0–5 on each of Accuracy, Universality, Progress, Theory, Completeness and Novelty. Requires a Solution Writeup with every artifact open-sourced.

## 4. Audit of our notebook (2026-10-17)

| Kernel | Internet | External inputs | Eligibility |
|---|---|---|---|
| `rochikarla/arc-prize-2026-baseline` (`kaggle_arc_baseline.ipynb`) | Off | Model `sorokin/qwen3_4b_grids15_sft139` (Transformers/bfloat16 v1, **Apache 2.0**). Kernel source `sorokin/pip-install-unsloth-flash-patch` (offline wheels). L4 machine | ⚠️ Eligible on paper, with the open-weights caveat below |

- ✅ Internet is off. The model is loaded from `/kaggle/input/models/...` and nothing is downloaded at runtime.
- ✅ The model weights are public under Apache 2.0, which is OSI-approved.
- ⚠️ **Open Source AI Definition.** Open weights alone are not enough. The fine-tune's **training code and data recipe** (sorokin's SFT on "grids15") must also be public. Find and link sorokin's public training notebook or repo, or fine-tune our own model from Qwen3 (Apache 2.0) with code we publish.
- ⚠️ The kernel is currently **private**. Before the prize review, publish the notebook (or a GitHub repo) under CC BY 4.0 and attach it to a Solution Writeup within 7 days of the deadline.
- ℹ️ The project docs still describe this notebook as a "CPU DSL solver". The current metadata runs it on a **GPU (L4)** with the Qwen3-4B model. Keep the docs and the metadata consistent.

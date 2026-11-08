"""Build kernel_nvarc_pooled/arc2_nvarc_pooled.ipynb from the vendored upstream notebook.

Upstream: woguoat, "ARC2 NVARC Full-Cost SOTA Triple Pass" (public Kaggle notebook, Apache 2.0,
public LB 31.94), which runs the open-sourced NVARC pipeline (ARC Prize 2025 winner; model
sorokin/qwen3_4b_grids15_sft139, Apache 2.0) with unchanged model maths and adds:
length-grouped DFS batches (~2x faster decoding), cheapest-first scheduling with per-task
budgets so all hidden tasks are processed, and a pooled second TTT pass with fresh seeds on
the least certain tasks in the leftover time.

This script keeps the code cells byte-identical and only replaces the header markdown with
our description, credits and eligibility note.

Usage:
    python tools/build_nvarc_pooled.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kernel_nvarc_pooled"
UPSTREAM = KERNEL_DIR / "upstream" / "woguoat_full_cost_pooled_v1.ipynb"
OUT = KERNEL_DIR / "arc2_nvarc_pooled.ipynb"

HEADER = """# ARC Prize 2026 — ARC-AGI-2: NVARC TTT, full-depth pass + pooled second pass

**Why.** Re-running the identical NVARC notebook scores anywhere between ~27 and ~32 on the public LB
(our own v5 resubmissions: 31.39 and 29.86): test-time training and batched DFS are stochastic, and the
stock configuration does not always reach every one of the 240 hidden tasks within 12 h. This version
keeps the NVARC model maths unchanged (128 TTT sequences, LoRA r=256, 16 augmented views, turbo DFS at
p>=0.2, 8-view augmentation rescoring, `score_kgmon` selection) and spends the compute better:

1. **Length-grouped DFS batches of up to 8 views** (~2x faster decoding on L4).
2. **Cheapest tasks first with per-task budgets** from an online-calibrated cost model, so every hidden
   task is processed.
3. **Leftover time re-runs the least certain tasks with fresh seeds** (second LoRA TTT run); candidates of
   both runs are pooled and vote together under `score_kgmon`.
4. Stable rescoring seeds, per-task error isolation, corrupt-file tolerant submission writer.

**Credits.** Code cells are taken unchanged from woguoat's public notebook
[ARC2 NVARC Full-Cost SOTA Triple Pass](https://www.kaggle.com/code/woguoat/arc2-nvarc-full-cost-sota-triple-pass)
(Apache 2.0), which builds on NVARC (Ivan Sorokin & Jean-François Puget, ARC Prize 2025 winner), the
ARChitects (Daniel Franzen & Jan Disselhoff) and mikelou1's throughput patch.

**Prize eligibility.** Model `sorokin/qwen3_4b_grids15_sft139` (Apache 2.0, NVARC's open-sourced
fine-tune), offline wheels from `sorokin/pip-install-unsloth-flash-patch`, code under Apache 2.0, to be
released under CC BY 4.0. Internet is off; everything is attached as Kaggle inputs.
"""


def build() -> None:
    nb = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    head = nb["cells"][0]
    assert head["cell_type"] == "markdown" and "full-depth single pass" in "".join(head["source"])
    head["source"] = HEADER.splitlines(keepends=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build()

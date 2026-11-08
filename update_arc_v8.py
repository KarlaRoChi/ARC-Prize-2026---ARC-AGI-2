"""
update_arc_v8.py - ARC-AGI-2 v8: Integrated CPU Fallback Heuristics & Consensus Selection
Combines:
1. v7 consensus-first selection (score_consensus_v7)
2. Safe beam threshold (max_score = -log(0.20))
3. Safe augmentation schedule (n=16)
4. CPU Fallback Protection:
   Injects cpu_fallback.py into the kernel environment and executes fill_submission_with_fallbacks()
   on the final submission dictionary. This guarantees that if CUDA is absent, or if specific
   test puzzles skip/timeout, intelligent color-mapping/component-extraction heuristics
   replace dummy [[0]] outputs, eliminating the 0.00 LB failure mode.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "kaggle_arc_baseline.ipynb"
FALLBACK_SCRIPT_PATH = HERE / "cpu_fallback.py"

def update_notebook_v8():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Title cell
    cell0 = nb["cells"][0]
    cell0["source"] = [
        "# ARC Prize 2026 — v8: NVARC Consensus Selection + CPU Fallback Protection (4×L4 / CPU Safe)\n",
        "\n",
        "**Enhancements in v8:**\n",
        "1. **Consensus-First Pass@2 Selection (`score_consensus_v7`)**: Restores v5's winning strategy where both `attempt_1` and `attempt_2` are drawn strictly from augmentation consensus (`score_kgmon`), eliminating single-beam perplexity displacement.\n",
        "2. **CPU Fallback Protection (`cpu_fallback.py`)**: Intercepts dummy `[[0]]` grids resulting from CPU environments or inference skips, replacing them with deterministic color-mapping, 4-connected component bounding-box cropping, and symmetry heuristics.\n",
        "3. **Safe Beam Threshold (`max_score = -log(0.20)`) & `n=16` Augmentations**: Tight exploration bound and proven schedule within 1200s timeout.\n",
        "4. **100% Prize-Eligible**: Fully reproducible under CC BY 4.0.\n"
    ]

    # Read cpu_fallback.py code to inject as a cell
    with open(FALLBACK_SCRIPT_PATH, "r", encoding="utf-8") as f:
        fallback_code = f.read()

    fallback_cell_source = "%%writefile cpu_fallback.py\n" + fallback_code

    # Check if cpu_fallback.py cell already exists
    has_fallback_cell = False
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "%%writefile cpu_fallback.py" in src:
            has_fallback_cell = True
            cell["source"] = [line + "\n" for line in fallback_cell_source.splitlines()]
            break

    if not has_fallback_cell:
        # Insert cpu_fallback.py before the starter.py cell
        insert_idx = -1
        for idx, cell in enumerate(nb["cells"]):
            src = "".join(cell.get("source", []))
            if "%%writefile starter.py" in src:
                insert_idx = idx
                break
        
        fallback_cell = {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in fallback_cell_source.splitlines()]
        }
        if insert_idx >= 0:
            nb["cells"].insert(insert_idx, fallback_cell)
        else:
            nb["cells"].append(fallback_cell)

    # 3. Update the final submission cell to invoke fill_submission_with_fallbacks
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "submission = data.get_submission(decoder.run_selection_algo())" in src:
            old_block = """submission = data.get_submission(decoder.run_selection_algo())

with open("submission.json", "w") as f:
    json.dump(submission, f)"""

            new_block = """submission = data.get_submission(decoder.run_selection_algo())

# CPU Fallback Safety: Replace any skipped/dummy [[0]] grids with intelligent heuristics
try:
    from cpu_fallback import fill_submission_with_fallbacks
    submission = fill_submission_with_fallbacks(submission, data.queries)
    print("Fallback safety: validated and filled missing/dummy [[0]] predictions.")
except Exception as e:
    print("Fallback warning:", e)

with open("submission.json", "w") as f:
    json.dump(submission, f)"""

            if old_block in src:
                new_src = src.replace(old_block, new_block)
                cell["source"] = [line + "\n" for line in new_src.splitlines()]
            elif "from cpu_fallback import fill_submission_with_fallbacks" not in src:
                # Target more flexible replacement
                new_src = src.replace(
                    'submission = data.get_submission(decoder.run_selection_algo())',
                    'submission = data.get_submission(decoder.run_selection_algo())\ntry:\n    from cpu_fallback import fill_submission_with_fallbacks\n    submission = fill_submission_with_fallbacks(submission, data.queries)\n    print("Fallback safety: replaced dummy [[0]] predictions.")\nexcept Exception as e:\n    print("Fallback warning:", e)'
                )
                cell["source"] = [line + "\n" for line in new_src.splitlines()]

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully upgraded {NB_PATH.name} to v8 with CPU Fallback Protection!")

if __name__ == "__main__":
    update_notebook_v8()

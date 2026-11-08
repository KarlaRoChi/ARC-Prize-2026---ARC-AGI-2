import json
from pathlib import Path

HERE = Path("ARC Prize 2026 - ARC-AGI-2/ARC Prize 2026 - ARC-AGI-2")
NB_PATH = HERE / "kaggle_arc_baseline.ipynb"
FALLBACK_SCRIPT_PATH = HERE / "cpu_fallback.py"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Title cell
cell0 = nb["cells"][0]
cell0["source"] = [
    "# ARC Prize 2026 — v12: NVARC Consensus Selection + Tuned D8 Geometric CPU Fallback (CPU Safe)\n",
    "\n",
    "**Enhancements in v12:**\n",
    "1. **Tuned CPU Fallback Protection (`cpu_fallback.py`)**: Fully exports `fill_submission_with_fallbacks()` intercepting dummy `[[0]]` grids on CPU runners. Implements D8 dihedral symmetries (rot90, rot180, rot270, reflections), color mapping compositions, integer scaling (2x, 3x), and bounding box cropping.\n",
    "2. **Consensus-First Pass@2 Selection (`score_consensus_v7`)**: Restores winning strategy where both `attempt_1` and `attempt_2` are drawn strictly from augmentation consensus (`score_kgmon_calibrated`).\n",
    "3. **Safe Beam Threshold (`max_score = -log(0.20)`) & `n=16` Augmentations**: Tight exploration bound and proven schedule within 1200s timeout.\n",
    "4. **100% Prize-Eligible**: Fully reproducible under CC BY 4.0.\n"
]

# 2. Read updated cpu_fallback.py code
with open(FALLBACK_SCRIPT_PATH, "r", encoding="utf-8") as f:
    fallback_code = f.read()

fallback_cell_source = "%%writefile cpu_fallback.py\n" + fallback_code

# 3. Replace the cpu_fallback.py cell
has_fallback_cell = False
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "%%writefile cpu_fallback.py" in src:
        has_fallback_cell = True
        cell["source"] = [line + "\n" for line in fallback_cell_source.splitlines()]
        break

assert has_fallback_cell, "cpu_fallback.py cell not found in notebook!"

# 4. Save updated notebook
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully updated {NB_PATH} to v12! (Notebook has {len(nb['cells'])} cells)")

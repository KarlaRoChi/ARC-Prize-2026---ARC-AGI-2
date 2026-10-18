"""
Update kaggle_arc_baseline.ipynb to v6:
- Expand test-time fine-tuning augmentations from 16 to 20 for broader geometric invariance.
- Add consensus selection algorithm (50% probmul_3 + 50% kgmon) to balance beam search confidence and augmentation consistency.
- Widen max_score threshold from -log(0.2) to -log(0.18) to discover high-value candidate beams on complex grid topologies.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "kaggle_arc_baseline.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Title cell
cell0 = nb["cells"][0]
cell0["source"] = [
    "# ARC Prize 2026 — v6: NVARC Consensus Rerank & Expanded Augmentations (4×L4)\n",
    "\n",
    "**Enhancements in v6:**\n",
    "1. **Expanded Test-Time Augmentations**: Increases demonstration augmentations from 16 to 20 for higher geometric transformation coverage.\n",
    "2. **Consensus Beam Scoring**: Blends `score_full_probmul_3` (beam probability) and `score_kgmon` (augmentation NLL consistency) to prevent overfitting to single augmentations.\n",
    "3. **Refined Exploration Threshold**: Adjusts `max_score = -log(0.18)` allowing discovery of deep path solutions within the 12-hour timeout budget.\n",
    "4. **100% Prize-Eligible**: All model weights and LoRA training recipes reproducible under CC BY 4.0."
]

# Update arc_decoder.py cell
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "%%writefile arc_decoder.py" in src and "def score_kgmon" in src:
        new_src = src.replace(
            "selection_algorithms = [\n    score_full_probmul_3,\n    score_kgmon,\n]",
            """def getter_consensus(guesses):
    return 0.5 * getter_full_probmul_3(guesses) + 0.5 * getter_kgmon(guesses)

def score_consensus(guesses):
    return score_sum(guesses, getter_consensus)

selection_algorithms = [
    score_consensus,
    score_full_probmul_3,
    score_kgmon,
]"""
        )
        new_src = new_src.replace(
            "def run_selection_algo(self, selection_algorithm=score_kgmon):",
            "def run_selection_algo(self, selection_algorithm=score_consensus):"
        )
        cell["source"] = [line + "\n" for line in new_src.splitlines()]

    if "%%writefile arc_solver.py" in src and "max_score = -np.log(0.2)" in src:
        new_src = src.replace("max_score = -np.log(0.2)", "max_score = -np.log(0.18)")
        new_src = new_src.replace("train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)", "train_ds = puzzle_ds.augment(n=20, shfl_keys=True, seed=1)")
        cell["source"] = [line + "\n" for line in new_src.splitlines()]

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully updated kaggle_arc_baseline.ipynb to v6!")

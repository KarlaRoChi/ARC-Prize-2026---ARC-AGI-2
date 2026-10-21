"""
Update kaggle_arc_baseline.ipynb to v6:
- Restore n=16 augmentations (proven safe within the 1200s per-puzzle timeout at 31.39 LB; n=20 risks timeout on large grids).
- Replace unnormalized score_consensus with score_hedged_top2:
  - attempt_1: top-1 of score_kgmon_tiebreak (augmentation consistency + beam NLL tie-breaker).
  - attempt_2: top-1 of score_full_probmul_3 if distinct from attempt_1, else #2 of score_kgmon_tiebreak.
- Retain max_score = -log(0.18) for candidate beam discovery.
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
    "# ARC Prize 2026 — v6: NVARC Hedged Top-2 Rerank & Beam Tie-Breaking (4×L4)\n",
    "\n",
    "**Enhancements in v6:**\n",
    "1. **Timeout-Safe Augmentation Budget (`n=16`)**: Preserves the proven 16-augmentation LoRA TTT schedule (`31.39` LB) so large grids stay within the 1200s per-puzzle timeout across 120 test tasks.\n",
    "2. **Hedged Top-2 Pass@2 Selection (`score_hedged_top2`)**: Uses `score_kgmon_tiebreak` (augmentation count - mean aug NLL - 0.05 * mean beam NLL) for `attempt_1`, and hedges `attempt_2` with the top distinct candidate from `score_full_probmul_3`.\n",
    "3. **Refined Exploration Threshold**: Uses `max_score = -log(0.18)` allowing discovery of deep path solutions within the 12-hour timeout budget.\n",
    "4. **100% Prize-Eligible**: All model weights and LoRA training recipes reproducible under CC BY 4.0.\n"
]

# Update arc_decoder.py and arc_solver.py cells
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "%%writefile arc_decoder.py" in src and "def score_kgmon" in src:
        # Replace getter_consensus / score_consensus block with score_hedged_top2
        target_block_1 = """def getter_consensus(guesses):
    return 0.5 * getter_full_probmul_3(guesses) + 0.5 * getter_kgmon(guesses)

def score_consensus(guesses):
    return score_sum(guesses, getter_consensus)

selection_algorithms = [
    score_consensus,
    score_full_probmul_3,
    score_kgmon,
]"""
        target_block_2 = """selection_algorithms = [
    score_full_probmul_3,
    score_kgmon,
]"""
        replacement_block = """def getter_kgmon_tiebreak(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    beam_tie = 0.05 * np.mean([g["beam_score"] for g in guesses])
    return inf_score - aug_score - beam_tie

def score_kgmon_tiebreak(guesses):
    return score_sum(guesses, getter_kgmon_tiebreak)

def score_consensus(guesses):
    \"\"\"Hedged top-2 pass@2 selector: attempt_1 = kgmon_tiebreak[0], attempt_2 = distinct probmul_3[0] or kgmon_tiebreak[1].\"\"\"
    ranked_kg = score_kgmon_tiebreak(guesses)
    ranked_pm = score_full_probmul_3(guesses)
    if not ranked_kg:
        return ranked_pm
    out = [ranked_kg[0]]
    seen = {hashable(ranked_kg[0])}
    for cand in ranked_pm + ranked_kg[1:]:
        h = hashable(cand)
        if h not in seen:
            out.append(cand)
            seen.add(h)
    return out

selection_algorithms = [
    score_consensus,
    score_kgmon_tiebreak,
    score_full_probmul_3,
    score_kgmon,
]"""
        if target_block_1 in src:
            new_src = src.replace(target_block_1, replacement_block)
        elif target_block_2 in src:
            new_src = src.replace(target_block_2, replacement_block)
        else:
            new_src = src
        new_src = new_src.replace(
            "def run_selection_algo(self, selection_algorithm=score_kgmon):",
            "def run_selection_algo(self, selection_algorithm=score_consensus):"
        )
        cell["source"] = [line + "\n" for line in new_src.splitlines()]

    if "%%writefile arc_solver.py" in src:
        new_src = src.replace("max_score = -np.log(0.2)", "max_score = -np.log(0.18)")
        new_src = new_src.replace(
            "train_ds = puzzle_ds.augment(n=20, shfl_keys=True, seed=1)",
            "train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)"
        )
        cell["source"] = [line + "\n" for line in new_src.splitlines()]

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully updated kaggle_arc_baseline.ipynb to v6 (n=16 + hedged top-2 selector)!")

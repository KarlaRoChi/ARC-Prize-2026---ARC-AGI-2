"""
selector_v8_clean.py - Clean consensus-preserving Pass@2 selector and diagnostic instrumentation.

100% Prize-Eligible (CC BY 4.0 compliant).
Resolves candidate displacement bug (ranked_pm[0] displacing ranked_kg[1]).

Provides:
- score_consensus_v8_clean: strict consensus Pass@2 selector with distinct fallback
- score_hedged_flawed_v7: diagnostic implementation of the flawed v7 selector
- score_kgmon: v5 baseline pure consensus selector
- SelectorDiagnosticTracker: collects selector comparison metrics on candidate pools
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import numpy as np


def hashable(guess: Any) -> Tuple[Tuple[int, ...], ...]:
    """Converts 2D numpy array or list of lists into hashable tuple of tuples."""
    if isinstance(guess, np.ndarray):
        return tuple(map(tuple, guess.tolist()))
    return tuple(map(tuple, guess))


def score_sum(guesses: Dict[str, Any], getter: Callable[[List[Dict[str, Any]]], float]) -> List[Any]:
    """Groups candidate solutions by equality, aggregates scores via getter, and sorts descending."""
    if not guesses:
        return []
    guess_list = list(guesses.values())
    scores: Dict[Tuple[Tuple[int, ...], ...], Tuple[List[Dict[str, Any]], Any]] = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores.setdefault(h, ([], g["solution"]))
        x[0].append(g)
    scored = [(getter(sc), o) for sc, o in scores.values()]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[-1] for x in scored]


# --- Baseline A: v5 Pure kgmon (LB 31.39) ---
def getter_kgmon(guesses: List[Dict[str, Any]]) -> float:
    inf_score = len(guesses)
    aug_score = float(np.mean([np.mean(g["score_aug"]) for g in guesses]))
    return inf_score - aug_score


def score_kgmon(guesses: Dict[str, Any]) -> List[Any]:
    return score_sum(guesses, getter_kgmon)


# --- Probability / Perplexity Scorer (full_probmul_3) ---
def getter_full_probmul_3(guesses: List[Dict[str, Any]], baseline: float = 3.0) -> float:
    inf_score = float(np.sum([baseline - g["beam_score"] for g in guesses]))
    aug_score = float(np.mean([np.sum([baseline - s for s in g["score_aug"]]) for g in guesses]))
    return inf_score + aug_score


def score_full_probmul_3(guesses: Dict[str, Any]) -> List[Any]:
    return score_sum(guesses, getter_full_probmul_3)


# --- Baseline B: v7 Flawed Hedged Selector ---
def getter_kgmon_tiebreak(guesses: List[Dict[str, Any]]) -> float:
    inf_score = len(guesses)
    aug_score = float(np.mean([np.mean(g["score_aug"]) for g in guesses]))
    beam_tie = 0.05 * float(np.mean([g["beam_score"] for g in guesses]))
    return inf_score - aug_score - beam_tie


def score_kgmon_tiebreak(guesses: Dict[str, Any]) -> List[Any]:
    return score_sum(guesses, getter_kgmon_tiebreak)


def score_hedged_flawed_v7(guesses: Dict[str, Any]) -> List[Any]:
    """The flawed v7 selector that forced ranked_pm[0] into attempt_2."""
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


# --- Candidate C: v8 Clean Consensus-Preserving Selector ---
def getter_kgmon_calibrated(guesses: List[Dict[str, Any]]) -> float:
    inf_score = len(guesses)
    aug_score = float(np.mean([np.mean(g["score_aug"]) for g in guesses]))
    beam_tie = 0.05 * float(np.mean([g["beam_score"] for g in guesses]))
    return inf_score - aug_score - beam_tie


def score_kgmon_calibrated(guesses: Dict[str, Any]) -> List[Any]:
    return score_sum(guesses, getter_kgmon_calibrated)


def score_consensus_v8_clean(guesses: Dict[str, Any]) -> List[Any]:
    """Clean consensus selector strictly preserving Top-2 consensus.
    attempt_1 = #1 candidate by augmentation consensus (score_kgmon_calibrated)
    attempt_2 = #2 candidate by augmentation consensus (score_kgmon_calibrated)
    Only falls back to distinct score_full_probmul_3 if fewer than 2 distinct consensus candidates exist.
    """
    if not guesses:
        return []
    ranked_kg = score_kgmon_calibrated(guesses)
    if len(ranked_kg) >= 2:
        return ranked_kg[:2]

    out = list(ranked_kg)
    seen = {hashable(c) for c in out}
    ranked_pm = score_full_probmul_3(guesses)
    for cand in ranked_pm:
        h = hashable(cand)
        if h not in seen:
            out.append(cand)
            seen.add(h)
        if len(out) >= 2:
            break
    return out


class SelectorDiagnosticTracker:
    """Lightweight diagnostics collector comparing selector outputs across puzzles.
    Logs ranking divergence without storing multi-gigabyte tensors.
    """

    def __init__(self) -> None:
        self.puzzle_stats: Dict[str, Dict[str, Any]] = {}

    def record_puzzle(self, base_key: str, guesses: Dict[str, Any]) -> Dict[str, Any]:
        if not guesses:
            stats = {
                "n_guesses": 0,
                "n_unique": 0,
                "v5_top2_hashes": [],
                "v7_top2_hashes": [],
                "v8_top2_hashes": [],
                "top1_diverged_v7_v8": False,
                "top2_diverged_v7_v8": False,
                "displaced_consensus_2": False,
            }
            self.puzzle_stats[base_key] = stats
            return stats

        ranked_v5 = score_kgmon(guesses)
        ranked_v7 = score_hedged_flawed_v7(guesses)
        ranked_v8 = score_consensus_v8_clean(guesses)

        h_v5 = [hashable(c) for c in ranked_v5[:2]]
        h_v7 = [hashable(c) for c in ranked_v7[:2]]
        h_v8 = [hashable(c) for c in ranked_v8[:2]]

        top1_div = (h_v7[0] != h_v8[0]) if (h_v7 and h_v8) else False
        top2_div = (len(h_v7) >= 2 and len(h_v8) >= 2 and h_v7[1] != h_v8[1])

        # Displacement occurs if v8 attempt_2 is consensus candidate #2, but v7 attempt_2 differs
        displaced = False
        if len(ranked_v5) >= 2 and len(ranked_v7) >= 2:
            if h_v7[1] != h_v5[1] and h_v8[1] == h_v5[1]:
                displaced = True

        stats = {
            "n_guesses": len(guesses),
            "n_unique": len(set(hashable(g["solution"]) for g in guesses.values())),
            "top1_diverged_v7_v8": top1_div,
            "top2_diverged_v7_v8": top2_div,
            "displaced_consensus_2": displaced,
        }
        self.puzzle_stats[base_key] = stats
        return stats

    def summary(self) -> Dict[str, Any]:
        n_puzzles = len(self.puzzle_stats)
        if n_puzzles == 0:
            return {"n_puzzles": 0}
        n_top1_div = sum(1 for s in self.puzzle_stats.values() if s["top1_diverged_v7_v8"])
        n_top2_div = sum(1 for s in self.puzzle_stats.values() if s["top2_diverged_v7_v8"])
        n_displaced = sum(1 for s in self.puzzle_stats.values() if s["displaced_consensus_2"])
        return {
            "n_puzzles_audited": n_puzzles,
            "top1_divergence_count": n_top1_div,
            "top1_divergence_pct": round(100.0 * n_top1_div / n_puzzles, 2),
            "top2_divergence_count": n_top2_div,
            "top2_divergence_pct": round(100.0 * n_top2_div / n_puzzles, 2),
            "consensus2_displaced_count": n_displaced,
            "consensus2_displaced_pct": round(100.0 * n_displaced / n_puzzles, 2),
        }

    def save(self, output_path: Path) -> None:
        data = {
            "summary": self.summary(),
            "puzzles": self.puzzle_stats,
        }
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

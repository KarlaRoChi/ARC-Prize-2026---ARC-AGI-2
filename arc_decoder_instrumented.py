"""
arc_decoder_instrumented.py - Isolated Instrumented ArcDecoder integrating clean Pass@2 selection.

Leaves baseline arc_decoder.py 100% untouched.
Provides lightweight diagnostics collection for future 4xL4 GPU runs.
"""

from __future__ import annotations

import bz2
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from selector_v8_clean import (
    SelectorDiagnosticTracker,
    hashable,
    score_consensus_v8_clean,
    score_hedged_flawed_v7,
    score_kgmon,
)


class InstrumentedArcDecoder:
    """ArcDecoder wrapper that records selector disagreement metrics during inference."""

    def __init__(self, dataset: Any = None, n_guesses: int = 2) -> None:
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results: Dict[str, Dict[str, Any]] = {}
        self.diagnostics = SelectorDiagnosticTracker()

    def load_decoded_results(self, store: str, run_name: str = "") -> None:
        if not os.path.exists(store):
            return
        for key in os.listdir(store):
            if not key.endswith(".bz2") and not key.endswith(".pkl"):
                continue
            p = os.path.join(store, key)
            try:
                if key.endswith(".bz2"):
                    with bz2.BZ2File(p) as f:
                        outputs = pickle.load(f)
                else:
                    with open(p, "rb") as f:
                        outputs = pickle.load(f)
            except Exception as e:
                print(f"Error loading {p}: {e}")
                continue

            base_key = key.split(".")[0]
            self.decoded_results.setdefault(base_key, {})
            for i, sample in enumerate(outputs):
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample

    def run_selection_algo(self, selection_algorithm: Any = score_consensus_v8_clean) -> Dict[str, List[Any]]:
        out = {}
        for bk, v in self.decoded_results.items():
            out[bk] = selection_algorithm(v)
            # Record diagnostic comparison
            self.diagnostics.record_puzzle(bk, v)
        return out

    def save_diagnostics(self, output_path: str = "selector_diagnostics.json") -> None:
        self.diagnostics.save(Path(output_path))
        print(f"Selector diagnostics saved to {output_path}: {self.diagnostics.summary()}")

# 🧩 Experiment Log: ARC Prize 2026

**Metric:** Exact Match Top-2 Accuracy per problem.  
**Resource:** 4×L4 GPU from v5 (1 GPU slot, 2× quota). v1–v4 ran on CPU.

| Version | Date | Hypothesis / Description | Technical Changes | Local CV | LB Score | Status / Conclusions |
|---|---|---|---|---|---|---|
| `v1_baseline` | 2026-10-15 | Base inference pipeline and `submission.json` format generation | Direct geometric heuristics + null-matrix fallbacks | N/A | `0.00` | Initial baseline deployed and working correctly |
| `v2_dsl_solver` | 2026-10-15 | Multi-heuristic DSL engine with rule search and structured fallbacks | D4 group, non-bg bbox crop, gravity up/down, compound color mapping, non-trivial fallbacks | N/A | `0.00` | Ran correctly. Low score due to limited rule coverage. |
| `v3_object_centric` | 2026-10-15 | **Confidence-scored rule engine** + 10 new rule families | Tiling `np.tile`, Kronecker, Grid-select, Fill-enclosed (scipy flood-fill), Mirror-complete H/V, Freq-rank remap, Fixed-output-size crop/fill, gravity L/R, per-task deadline timeout (10s) | N/A | `0.00` | Rules scored continuously (fraction of train pairs matched); top-2 → attempt_1/attempt_2. Broader coverage of frequent ARC patterns. |
| `v4_subgrid_object_algebra` | 2026-10-15 | **Subgrid Partition Logic & Object Extractor** for complex compositional tasks | Detection of single-color horizontal/vertical divider lines (`detect_dividers`) with logical overlay (`np.where(B != 0, B, T)`), extraction of the connected component with maximum area (`extract_largest_object`) and minimum area (`extract_smallest_object`). Strict assignment of 2 hypotheses per test. | 0.080 | `0.00` | Solves grid-overlay and salient-object-isolation tasks in the validation dataset without exceeding the CPU compute budget. |
| `v5_nvarc_qwen3_ttt` | 2026-10-17 | **Paradigm shift: CPU DSL → NVARC (LLM + test-time fine-tuning).** Four DSL versions scored 0.00; ARC-AGI-2 cannot be solved with hand-crafted primitives. | Qwen3-4B `sorokin/qwen3_4b_grids15_sft139` offline, LoRA r=256 trained per task with 16 D4+color augmentations, DFS decoding over 12 ARC tokens, NLL re-ranking with 8 augmentations, 4 workers on 4×L4, 12 h budget. Adapted from `mikelou1/arc-agi2-lb33-89-minimal-perfpatch`. | Commit run (4 eval tasks): 2.5/4 tasks (3/5 test inputs) | Pending submission | Public reference: LB 33.89. Target: >30. |

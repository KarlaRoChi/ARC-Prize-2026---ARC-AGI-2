"""
evaluate_arc2.py - Independent Official Metric Evaluator for ARC-AGI-2

Implements the exact official competition scoring definition:
- Individual test outputs are scored pass@2 (attempt_1 or attempt_2 matches ground truth).
- Aggregate score = (correct test outputs) / (total test outputs).
- Also reports fully solved tasks (all test outputs for a task correct).
- Audits and classifies errors: missing tasks, invalid grids, malformed attempts, shape mismatches.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple


def is_valid_grid(grid: Any) -> bool:
    """Validate that grid is a 2D list of integers [0..9]."""
    if not isinstance(grid, list) or len(grid) == 0:
        return False
    if not all(isinstance(row, list) and len(row) > 0 for row in grid):
        return False
    w = len(grid[0])
    for row in grid:
        if len(row) != w:
            return False
        for val in row:
            if not isinstance(val, int) or val < 0 or val > 9:
                return False
    return True


def evaluate_submission(
    submission_path: str,
    solutions_path: str,
    challenges_path: str = None,
    verbose: bool = True
) -> Dict[str, Any]:
    with open(submission_path, "r", encoding="utf-8") as f:
        sub = json.load(f)
    with open(solutions_path, "r", encoding="utf-8") as f:
        sol = json.load(f)
        
    challenges = {}
    if challenges_path and Path(challenges_path).exists():
        with open(challenges_path, "r", encoding="utf-8") as f:
            challenges = json.load(f)

    total_tasks = len(sol)
    total_test_outputs = sum(len(pairs) for pairs in sol.values())
    
    correct_test_outputs = 0
    fully_solved_tasks = 0
    
    missing_tasks = []
    output_count_mismatches = []
    malformed_attempts = []
    shape_mismatches = []
    value_mismatches = []
    
    per_task_details = {}

    for tid, gt_outputs in sol.items():
        if tid not in sub:
            missing_tasks.append(tid)
            per_task_details[tid] = {
                "status": "MISSING_TASK",
                "outputs_correct": 0,
                "total_outputs": len(gt_outputs),
                "fully_solved": False
            }
            continue

        pred_outputs = sub[tid]
        if not isinstance(pred_outputs, list) or len(pred_outputs) != len(gt_outputs):
            output_count_mismatches.append((tid, len(gt_outputs), len(pred_outputs) if isinstance(pred_outputs, list) else -1))
            per_task_details[tid] = {
                "status": "OUTPUT_COUNT_MISMATCH",
                "expected": len(gt_outputs),
                "got": len(pred_outputs) if isinstance(pred_outputs, list) else -1,
                "outputs_correct": 0,
                "total_outputs": len(gt_outputs),
                "fully_solved": False
            }
            continue

        task_all_correct = True
        task_correct_count = 0
        output_results = []

        for out_idx, (gt_grid, pred_dict) in enumerate(zip(gt_outputs, pred_outputs)):
            if not isinstance(pred_dict, dict):
                malformed_attempts.append((tid, out_idx, "not a dictionary"))
                task_all_correct = False
                continue

            a1 = pred_dict.get("attempt_1")
            a2 = pred_dict.get("attempt_2")

            a1_valid = is_valid_grid(a1)
            a2_valid = is_valid_grid(a2)

            if not a1_valid:
                malformed_attempts.append((tid, out_idx, "attempt_1 invalid grid"))
            if not a2_valid:
                malformed_attempts.append((tid, out_idx, "attempt_2 invalid grid"))

            gt_shape = (len(gt_grid), len(gt_grid[0]))
            a1_shape = (len(a1), len(a1[0])) if a1_valid else None
            a2_shape = (len(a2), len(a2[0])) if a2_valid else None

            m1 = (a1 == gt_grid)
            m2 = (a2 == gt_grid)
            solved_output = m1 or m2

            if solved_output:
                correct_test_outputs += 1
                task_correct_count += 1
            else:
                task_all_correct = False
                if a1_shape != gt_shape and a2_shape != gt_shape:
                    shape_mismatches.append((tid, out_idx, gt_shape, a1_shape, a2_shape))
                else:
                    value_mismatches.append((tid, out_idx))

            output_results.append({
                "output_idx": out_idx,
                "solved": solved_output,
                "attempt_1_match": m1,
                "attempt_2_match": m2,
                "a1_shape": a1_shape,
                "a2_shape": a2_shape,
                "gt_shape": gt_shape
            })

        if task_all_correct and len(gt_outputs) > 0:
            fully_solved_tasks += 1

        per_task_details[tid] = {
            "status": "SOLVED" if task_all_correct else "PARTIAL" if task_correct_count > 0 else "FAILED",
            "outputs_correct": task_correct_count,
            "total_outputs": len(gt_outputs),
            "fully_solved": task_all_correct,
            "outputs": output_results
        }

    output_acc = (correct_test_outputs / total_test_outputs) * 100.0 if total_test_outputs > 0 else 0.0
    task_acc = (fully_solved_tasks / total_tasks) * 100.0 if total_tasks > 0 else 0.0

    report = {
        "summary": {
            "total_tasks": total_tasks,
            "total_test_outputs": total_test_outputs,
            "correct_test_outputs": correct_test_outputs,
            "output_level_accuracy_pct": round(output_acc, 4),
            "fully_solved_tasks": fully_solved_tasks,
            "task_level_accuracy_pct": round(task_acc, 4),
            "missing_tasks_count": len(missing_tasks),
            "output_count_mismatches_count": len(output_count_mismatches),
            "malformed_attempts_count": len(malformed_attempts),
            "shape_mismatches_count": len(shape_mismatches),
        },
        "details": per_task_details
    }

    if verbose:
        print("=" * 70)
        print("ARC-AGI-2 INDEPENDENT OFFICIAL METRIC EVALUATION")
        print("=" * 70)
        print(f"Total Tasks in Benchmark:           {total_tasks}")
        print(f"Total Test Outputs:                 {total_test_outputs}")
        print(f"Correct Test Outputs (Pass@2):      {correct_test_outputs} / {total_test_outputs} ({output_acc:.2f}%)")
        print(f"Fully Solved Tasks (All Outputs):   {fully_solved_tasks} / {total_tasks} ({task_acc:.2f}%)")
        print("-" * 70)
        print(f"Missing Tasks:                      {len(missing_tasks)}")
        print(f"Output Count Mismatches:            {len(output_count_mismatches)}")
        print(f"Malformed Attempts:                 {len(malformed_attempts)}")
        print(f"Shape Mismatches (both attempts):   {len(shape_mismatches)}")
        print("=" * 70)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ARC-AGI-2 submission against ground truth.")
    parser.add_argument("submission", type=str, help="Path to submission.json")
    parser.add_argument("solutions", type=str, help="Path to solutions.json")
    parser.add_argument("--challenges", type=str, default=None, help="Path to challenges.json")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation summary JSON")
    args = parser.parse_args()

    res = evaluate_submission(args.submission, args.solutions, args.challenges)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved evaluation report to {args.output}")

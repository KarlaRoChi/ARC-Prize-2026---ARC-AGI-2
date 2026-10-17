# 📊 ARC Prize 2026 — Summary of the Kaggle Dataset Description

**Competition:** [ARC Prize 2026 - ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)

---

## 1. 📁 Dataset Composition and Structure
The ARC-AGI-2 dataset provides an expanded human calibration designed to challenge the most advanced reasoning systems:

| File | Count | Description | Purpose |
|---|---|---|---|
| **`arc-agi_training_challenges.json`** | 1,000 tasks | Public tasks of varying difficulty (easy to complex) | Local training, DSL induction and primitive definition |
| **`arc-agi_training_solutions.json`** | 1,000 tasks | Full solutions for the training set | Direct supervision and cross-validation |
| **`arc-agi_evaluation_challenges.json`** | 120 tasks | Unseen public tasks | Local validation benchmark (OOD evaluation) |
| **`arc-agi_evaluation_solutions.json`** | 120 tasks | Solutions for the public evaluation set | Rigorous local CV measurement |
| **`arc-agi_test_challenges.json`** | Placeholder (copy of evaluation tasks); **240 hidden tasks** at rerun | Swapped for the real test set when the notebook is re-run | Final competition inference |
| **`sample_submission.json`** | Template | Sample of the expected JSON structure with 2 attempts per test pair | Schema validation |

---

## 2. 🧩 Internal Data Format and Grid Encoding
Each JSON file contains a dictionary whose keys are alphanumeric task identifiers (`task_id`, e.g. `"007bbfb7"`).

Each task contains two lists:
- `"train"`: List of 2 to 5 demonstration pairs `{"input": [[...]], "output": [[...]]}`.
- `"test"`: List of 1 to 3 test problems `{"input": [[...]]}` (and optionally `"output"` in sets that include solutions).

### Dimensions and Colors:
- **Dimensions:** Matrices range from $1 \times 1$ to $30 \times 30$. Output dimensions can be equal to, a divisor of, or a multiple of the input dimensions.
- **Pixel Values (10 discrete colors):**
  - `0`: Black (usually background / neutral)
  - `1`: Blue
  - `2`: Red
  - `3`: Green
  - `4`: Yellow
  - `5`: Gray
  - `6`: Magenta
  - `7`: Orange
  - `8`: Cyan / Teal
  - `9`: Brown

---

## 3. 📄 Strict Submission File Schema (`submission.json`)
The submission must be a valid JSON file with the following exact structure:

```json
{
  "00576224": [
    {
      "attempt_1": [[0, 0], [0, 0]],
      "attempt_2": [[0, 0], [0, 0]]
    }
  ],
  "007bbfb7": [
    {
      "attempt_1": [[1, 2], [3, 4]],
      "attempt_2": [[1, 1], [1, 1]]
    }
  ]
}
```

### Critical Evaluator Rules:
1. Each attempt must be a homogeneous 2D integer matrix with values in the range $[0, 9]$.
2. If a task contains multiple test problems in `"test"`, the list must contain one dictionary per problem, in the corresponding order.
3. There must be no null values (`NaN`), floats or ragged dimensions.

---

## 4. ℹ️ Notes from the official data page
- Most of the 240 hidden tasks have **one** test input. A small number have **two**, and predictions must follow the order of the test inputs.
- Every `task_id` in the challenges file must appear in `submission.json`, and both `attempt_1` and `attempt_2` must be present even if they are identical.
- Data license: **Apache 2.0**.

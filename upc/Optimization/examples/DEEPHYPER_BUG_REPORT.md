# Bug Report: `pareto_efficient` is all-False when multi-objective run-function returns integers

## Description

When a multi-objective run-function returns Python `int` values (which become `numpy.int64` in the results DataFrame), `compute_pareto_efficiency()` silently produces an all-False `pareto_efficient` column — even when non-dominated solutions clearly exist. Returning `float` for the same numerical values produces correct results.

## Root Cause

The issue is in `deephyper/analysis/hpo/_hpo.py`, function `get_mask_of_rows_without_failures()`:

```python
def get_mask_of_rows_without_failures(df, column):
    if pd.api.types.is_string_dtype(df[column]):
        mask_no_failures = ~df[column].str.startswith("F").to_numpy()
    else:
        mask_no_failures = df[column].map(lambda x: isinstance(x, float)).to_numpy()
        #                                           ^^^^^^^^^^^^^^^^^^^^^^^^
        #  BUG: numpy.int64 is NOT a subclass of Python float
```

When the objective column has `int64` dtype, every element is `numpy.int64`, and `isinstance(np.int64(v), float)` returns `False`. This misclassifies **every row** as a "failure", so `compute_pareto_efficiency()` has zero valid points and the Pareto front is empty.

The same function is also used by `filter_failed_objectives()`, so analysis utilities like `parameters_at_max()` and `plot_search_trajectory_single_objective_hpo()` are also affected.

## Suggested Fix

Replace the `isinstance(x, float)` check with something that accepts all numeric types:

```python
# Option A: check for numeric types (recommended)
else:
    mask_no_failures = df[column].map(
        lambda x: isinstance(x, (int, float, np.integer, np.floating))
    ).to_numpy()

# Option B: use pandas/numpy dtype checks
else:
    if pd.api.types.is_numeric_dtype(df[column]):
        mask_no_failures = np.ones(len(df), dtype=bool)
    else:
        mask_no_failures = df[column].map(lambda x: isinstance(x, float)).to_numpy()
```

## Minimal Reproduction

```python
#!/usr/bin/env python3
"""Minimal reproduction of the pareto_efficient int64 bug."""

import numpy as np
import pandas as pd
from deephyper.hpo import CBO, HpProblem
from deephyper.evaluator import Evaluator
from deephyper.analysis.hpo._hpo import get_mask_of_rows_without_failures

# --- Direct demonstration of the isinstance bug ---
df_float = pd.DataFrame({"objective_0": [1.0, 2.0, 3.0]})
_, mask_f = get_mask_of_rows_without_failures(df_float, "objective_0")
print(f"float64 column → mask = {mask_f}")   # [True  True  True]  ← correct

df_int = pd.DataFrame({"objective_0": [1, 2, 3]})
_, mask_i = get_mask_of_rows_without_failures(df_int, "objective_0")
print(f"int64   column → mask = {mask_i}")    # [False False False] ← BUG

# --- Full CBO search ---
def run_int(job):
    rng = np.random.RandomState(abs(hash(tuple(sorted(job.items())))) % 2**31)
    return (-int(rng.randint(1000, 100000)), -int(rng.randint(100, 10000)))

def run_float(job):
    rng = np.random.RandomState(abs(hash(tuple(sorted(job.items())))) % 2**31)
    return (-float(rng.randint(1000, 100000)), -float(rng.randint(100, 10000)))

problem = HpProblem()
problem.add_hyperparameter((1, 64), "x")
problem.add_hyperparameter((1, 64), "y")

for label, fn in [("int", run_int), ("float", run_float)]:
    search = CBO(problem, random_state=42, log_dir=f"/tmp/dh_bug_{label}", verbose=0)
    evaluator = Evaluator.create(fn, method="thread", method_kwargs={"num_workers": 1})
    results = search.search(evaluator, max_evals=20)
    print(f"{label:>5}: dtype={results['objective_0'].dtype}, "
          f"pareto_efficient={results['pareto_efficient'].sum()}/{len(results)}")
```

**Output:**
```
float64 column → mask = [ True  True  True]
int64   column → mask = [False False False]
  int: dtype=int64, pareto_efficient=0/20
float: dtype=float64, pareto_efficient=4/20
```

## Environment

- deephyper 0.13.2 (also confirmed on latest `master`)
- Python 3.11.11
- numpy 2.4.2
- pandas 3.0.1
- OS: Linux 6.17.0

## Workaround

Cast all objective values to `float()` before returning from the run-function:

```python
def run(job):
    obj1 = compute_objective_1(job)
    obj2 = compute_objective_2(job)
    return (float(obj1), float(obj2))  # always float, never int
```

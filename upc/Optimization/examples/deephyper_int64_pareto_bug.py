#!/usr/bin/env python3
"""
Minimal reproduction: DeepHyper pareto_efficient is all-False when
multi-objective run-functions return Python ints (numpy int64 in DataFrame).

BUG LOCATION:
    deephyper/analysis/hpo/_hpo.py  →  get_mask_of_rows_without_failures()

    Line:
        mask_no_failures = df[column].map(lambda x: isinstance(x, float)).to_numpy()

    Problem:
        numpy.int64 is NOT a subclass of Python float, so isinstance(np.int64(v), float)
        returns False.  When all objectives are integers, EVERY row is misclassified as
        a "failure" and compute_pareto_efficiency() has zero valid points to process,
        resulting in pareto_efficient being all-False.

    Suggested fix:
        Use  isinstance(x, (int, float, np.integer, np.floating))
        or   np.issubdtype(df[column].dtype, np.number) and not is_string_dtype(...)

IMPACT:
    Any multi-objective search whose run-function returns Python ints will
    silently produce an all-False pareto_efficient column — even though
    non-dominated solutions clearly exist.

Environment tested:
    deephyper==0.13.2  (also confirmed on latest master)
    Python 3.11.11, numpy 2.4.2, pandas 3.0.1

Usage:
    python deephyper_int64_pareto_bug.py
"""

import sys
import numpy as np
import pandas as pd

print("=" * 60)
print("DeepHyper pareto_efficient bug — isinstance(int64, float)")
print("=" * 60)

try:
    import deephyper
    print(f"  deephyper: {deephyper.__version__}")
except ImportError:
    print("  ERROR: deephyper not installed"); sys.exit(1)
print(f"  numpy:     {np.__version__}")
print(f"  pandas:    {pd.__version__}")
print(f"  python:    {sys.version.split()[0]}")
print()

# ─── Step 1: Show the isinstance() issue directly ───────────────────────
print("STEP 1: isinstance check on numpy scalars")
print(f"  isinstance(np.float64(1.0), float) = {isinstance(np.float64(1.0), float)}")
print(f"  isinstance(np.int64(1),     float) = {isinstance(np.int64(1), float)}")
print()

# ─── Step 2: Show get_mask_of_rows_without_failures misbehaves ──────────
from deephyper.analysis.hpo._hpo import get_mask_of_rows_without_failures

print("STEP 2: get_mask_of_rows_without_failures() on int64 vs float64 columns")

df_float = pd.DataFrame({"objective_0": [1.0, 2.0, 3.0]})
_, mask_f = get_mask_of_rows_without_failures(df_float, "objective_0")
print(f"  float64 column → mask = {mask_f}  (all True  ← correct)")

df_int = pd.DataFrame({"objective_0": [1, 2, 3]})
_, mask_i = get_mask_of_rows_without_failures(df_int, "objective_0")
print(f"  int64   column → mask = {mask_i}  (all False ← BUG)")
print()

# ─── Step 3: Full CBO search showing the effect ─────────────────────────
print("STEP 3: Full CBO search — int vs float objectives")
print("-" * 60)

from deephyper.hpo import CBO, HpProblem
from deephyper.evaluator import Evaluator

def run_int(job):
    """Return objectives as Python ints → int64 in DataFrame."""
    rng = np.random.RandomState(abs(hash(tuple(sorted(job.items())))) % 2**31)
    t = rng.randint(1_000, 100_000)
    b = rng.randint(100, 10_000)
    return (-int(t), -int(b))

def run_float(job):
    """Return objectives as Python floats → float64 in DataFrame."""
    rng = np.random.RandomState(abs(hash(tuple(sorted(job.items())))) % 2**31)
    t = rng.randint(1_000, 100_000)
    b = rng.randint(100, 10_000)
    return (-float(t), -float(b))

problem = HpProblem()
problem.add_hyperparameter((1, 64), "x")
problem.add_hyperparameter((1, 64), "y")

BUDGET = 20
SEED = 42

for label, fn in [("int", run_int), ("float", run_float)]:
    search = CBO(problem, random_state=SEED, log_dir=f"./tmp/dh_bug_{label}", verbose=0)
    evaluator = Evaluator.create(fn, method="thread", method_kwargs={"num_workers": 1})
    results = search.search(evaluator, max_evals=BUDGET)

    if results is not None:
        obj_dtype = results["objective_0"].dtype
        dh_pareto = int(results["pareto_efficient"].sum())
        total = len(results)
        print(f"  {label:>5} objectives: dtype={obj_dtype}, "
              f"pareto_efficient={dh_pareto}/{total}")
    else:
        print(f"  {label:>5} objectives: search returned None")

print()
print("EXPECTED: Both should report pareto_efficient > 0")
print("ACTUAL:   int objectives → pareto_efficient = 0  (all rows treated as 'failures')")
print()
print("Root cause: deephyper/analysis/hpo/_hpo.py")
print("  get_mask_of_rows_without_failures() line:")
print("    mask_no_failures = df[column].map(lambda x: isinstance(x, float)).to_numpy()")
print("  numpy.int64 is not a subclass of float → every int row = 'failure'")

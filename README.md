# MPS Benchmark Framework

This repository provides a lightweight framework to reproduce three experimental lines:

1. **Scale benchmark (D=16→128)** for Algorithm 2 vs brute-force and Schur baselines.
2. **Complex LCL formulas** (five templates) for validation in Section 8-style narratives.
3. **Model catalog** with 5 physical + 5 synthetic cases (D≤16) mapped to non-trivial LCL formulas.

## Structure

- `benchmark/algorithm2.py`: Reference Algorithm 2 implementation.
- `benchmark/random_channels.py`: Random block-structured channel generators with controlled periods.
- `benchmark/predicates.py`: Interval predicate helpers for adaptive thresholds.
- `benchmark/experiments/scale_benchmark.py`: Benchmark orchestration logic.
- `benchmark/experiments/lcl_formulas.py`: Five complex LCL formula templates.
- `benchmark/experiments/model_catalog.py`: Ten model cases (physical + synthetic).
- `scripts/run_scale_benchmark.py`: CLI entrypoint for scale experiments.

## Running the scale benchmark

```bash
python scripts/run_scale_benchmark.py
```

The runner prints a JSON summary and can be extended to serialize full metrics into CSV or parquet.

## Using the three experiment lines

### 1) Scale benchmark (D=16→128)

Run the full sweep (D values, eps values, seeds, repeats) with Algorithm 2 vs baselines:

```bash
python scripts/run_scale_benchmark.py
```

The benchmark wiring is in `benchmark/experiments/scale_benchmark.py`, including timing splits
and agreement metrics. You can edit the `BenchmarkConfig` in the script to reduce the sweep
for quick dry runs.【F:benchmark/experiments/scale_benchmark.py†L16-L109】

### 2) Complex LCL formula templates

List the five predefined formula templates (with predicates + descriptions) for the
“complex formula verification” section:

```bash
python scripts/list_lcl_formulas.py
```

The templates live in `benchmark/experiments/lcl_formulas.py` and can be extended
for additional physics scenarios.【F:benchmark/experiments/lcl_formulas.py†L1-L48】

### 3) 10 model verification catalog (5 physical + 5 synthetic)

Print the predefined model catalog (bond dimension constraints + recommended formulas):

```bash
python scripts/list_models.py
```

The catalog is defined in `benchmark/experiments/model_catalog.py` and can be adapted
to match your own MPS families or parameter scans.【F:benchmark/experiments/model_catalog.py†L1-L102】

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

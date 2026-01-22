"""CLI entrypoint for scale benchmark."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.experiments.scale_benchmark import BenchmarkConfig, run_scale_benchmark


def main() -> None:
    config = BenchmarkConfig(
        D_values=[16, 32, 64, 96, 128],
        eps_values=[0.0, 0.05, 0.1],
        seeds=list(range(10)),
        repeats=3,
        N_max=240,
        tail_window=12,
    )
    results = run_scale_benchmark(config)
    payload = {
        "count": len(results),
        "keys": [str(k) for k in results.keys()][:5],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

"""Print available LCL formula templates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.experiments.lcl_formulas import FORMULAS


def main() -> None:
    payload = [
        {
            "name": f.name,
            "formula": f.formula,
            "predicates": f.predicates,
            "description": f.description,
        }
        for f in FORMULAS
    ]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

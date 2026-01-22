"""Print physical and synthetic model catalogs."""
from __future__ import annotations

import json

from benchmark.experiments.model_catalog import ARTIFICIAL_MODELS, PHYSICAL_MODELS


def _serialize(model):
    return {
        "name": model.name,
        "kind": model.kind,
        "bond_dim": model.bond_dim,
        "description": model.description,
        "formula": model.formula,
        "metrics": model.metrics,
    }


def main() -> None:
    payload = {
        "physical": [_serialize(m) for m in PHYSICAL_MODELS],
        "synthetic": [_serialize(m) for m in ARTIFICIAL_MODELS],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

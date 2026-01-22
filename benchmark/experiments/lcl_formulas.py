"""Library of complex LCL formula templates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class LCLFormula:
    name: str
    formula: str
    predicates: List[str]
    description: str


FORMULAS = [
    LCLFormula(
        name="F1-energy-density",
        formula=r"\Phi_1 := \ell_{highE} \wedge E\,G\,\ell_{lowE}",
        predicates=["highE", "lowE"],
        description=(
            "Energy density relaxes from high boundary values to a stable low-energy window.")
    ),
    LCLFormula(
        name="F2-gap-stability",
        formula=r"\Phi_2 := G(\neg \ell_{gap\downarrow} \vee X \neg \ell_{gap\downarrow})",
        predicates=["gap_down"],
        description="Gap does not collapse as chain length increases.",
    ),
    LCLFormula(
        name="F3-correlation-clustering",
        formula=r"\Phi_3 := E\,G\,\ell_{corr\downarrow(d)}",
        predicates=["corr_down"],
        description="Correlations at fixed distance decay and stay small after some N.",
    ),
    LCLFormula(
        name="F4-cluster-stabilizer",
        formula=r"\Phi_4 := G\,\ell_{stab}",
        predicates=["stab"],
        description="All stabilizer expectations remain near +1 across chain lengths.",
    ),
    LCLFormula(
        name="F5-quantum-walk-advantage",
        formula=r"\Phi_5 := E\,G\,\ell_{adv}",
        predicates=["adv"],
        description="Quantum walk advantage eventually appears and persists.",
    ),
]

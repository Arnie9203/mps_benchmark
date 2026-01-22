"""Model catalog for physical and synthetic verification cases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ModelCase:
    name: str
    kind: str
    bond_dim: int
    description: str
    formula: str
    metrics: List[str]


PHYSICAL_MODELS = [
    ModelCase(
        name="AKLT",
        kind="physical",
        bond_dim=2,
        description="Gapped spin-1 AKLT chain with short-range correlations.",
        formula="F3-correlation-clustering or F2-gap-stability",
        metrics=["two-point correlations", "gap estimates"],
    ),
    ModelCase(
        name="Cluster",
        kind="physical",
        bond_dim=2,
        description="1D cluster stabilizer state for MBQC.",
        formula="F4-cluster-stabilizer",
        metrics=["stabilizer expectations", "noise threshold"],
    ),
    ModelCase(
        name="GHZ",
        kind="physical",
        bond_dim=2,
        description="Cat state with long-range order but zero local magnetization.",
        formula="Custom GHZ formula",
        metrics=["local magnetization", "end-to-end correlation"],
    ),
    ModelCase(
        name="Fredkin",
        kind="physical",
        bond_dim=16,
        description="Dyck-path state with slow correlation decay.",
        formula="F3-correlation-clustering (expect failure)",
        metrics=["fixed-distance correlations", "counterexample N"],
    ),
    ModelCase(
        name="Motzkin",
        kind="physical",
        bond_dim=16,
        description="Motzkin path state with non-local structure.",
        formula="F3-correlation-clustering or flatness formula",
        metrics=["local distribution flatness", "correlation rebound"],
    ),
]

ARTIFICIAL_MODELS = [
    ModelCase(
        name="Short-range TI-MPS",
        kind="synthetic",
        bond_dim=8,
        description="Random TI-MPS with small second eigenvalue to mimic gapped phase.",
        formula="F3-correlation-clustering",
        metrics=["spectral gap", "correlation length"],
    ),
    ModelCase(
        name="Near-critical TI-MPS",
        kind="synthetic",
        bond_dim=8,
        description="Random TI-MPS with |lambda2|≈0.99 to mimic criticality.",
        formula="F3-correlation-clustering (expect failure)",
        metrics=["lambda2", "counterexample N"],
    ),
    ModelCase(
        name="Multi-sector cat MPS",
        kind="synthetic",
        bond_dim=5,
        description="Superposition of classical sectors with multiple long-range orders.",
        formula="Custom multi-predicate formula",
        metrics=["uniform magnetization", "alternating correlations"],
    ),
    ModelCase(
        name="Oscillatory correlations",
        kind="synthetic",
        bond_dim=8,
        description="Complex eigenvalues induce oscillatory correlation decay.",
        formula="Oscillatory sign predicate",
        metrics=["odd/even correlations", "period"],
    ),
    ModelCase(
        name="Noisy random MPS",
        kind="synthetic",
        bond_dim=16,
        description="Short-range MPS perturbed by local noise to test robustness.",
        formula="Stabilizer threshold formula",
        metrics=["noise strength", "failure threshold"],
    ),
]

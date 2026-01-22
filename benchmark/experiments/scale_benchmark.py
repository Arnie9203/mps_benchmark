"""Scale benchmark wiring for Algorithm 2 vs baselines."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import numpy.linalg as LA

from benchmark.algorithm2 import algorithm2_from_Aks
from benchmark.predicates import IntervalPredicate, adaptive_interval
from benchmark.random_channels import build_block_channel, mix_with_identity


@dataclass
class BenchmarkConfig:
    D_values: List[int]
    eps_values: List[float]
    seeds: List[int]
    repeats: int
    N_max: int
    tail_window: int


@dataclass
class BenchmarkResult:
    runtime: Dict[str, float]
    memory: Dict[str, float]
    metrics: Dict[str, float]
    info: Dict[str, object]


def _gamma_from_superop(kraus: List[np.ndarray]) -> np.ndarray:
    d = kraus[0].shape[0]
    M = np.zeros((d * d, d * d), dtype=complex)
    for A in kraus:
        M += np.kron(A.conj(), A)
    return M


def _eval_predicate(gamma: callable, pred: IntervalPredicate, N_max: int) -> List[bool]:
    return [pred(gamma(n)) for n in range(1, N_max + 1)]


def _tail_eventually(truths: List[bool], tail_window: int) -> Tuple[set[int], set[int]]:
    omega_plus = set()
    omega_minus = set()
    N_max = len(truths)
    for i in range(tail_window, N_max + 1):
        if all(truths[i - tail_window : i]):
            omega_plus.add(i)
        if not any(truths[i - tail_window : i]):
            omega_minus.add(i)
    return omega_plus, omega_minus


def _alg2_predicate(
    omega_plus: set[int],
    omega_minus: dict[int, dict[str, object]],
    kappa: int,
    N: int,
) -> bool:
    i = N % kappa
    if i not in omega_plus:
        return False
    info = omega_minus.get(i)
    if info is None:
        return False
    Ni = info.get("Ni")
    exceptions = set(info.get("exceptions", []))
    if Ni is None:
        return False
    if N < Ni:
        return N in exceptions
    return True


def _time_and_memory(fn):
    import tracemalloc

    tracemalloc.start()
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


def run_scale_benchmark(config: BenchmarkConfig) -> Dict[Tuple[int, float, int, int], BenchmarkResult]:
    results = {}
    for D in config.D_values:
        for eps in config.eps_values:
            for seed in config.seeds:
                for repeat in range(config.repeats):
                    rng = np.random.default_rng(seed + repeat * 100)
                    family = build_block_channel(rng, D=D, d=3, periods=[1, 2])
                    kraus = mix_with_identity(family.kraus, eps)

                    M = _gamma_from_superop(kraus)
                    eigvals = LA.eigvals(M)
                    gamma = lambda N: float(np.sum(eigvals ** N).real)

                    sample_vals = [gamma(n) for n in range(20, 41)]
                    interval = adaptive_interval(sample_vals, width=0.05)

                    alg2, t_alg2, mem_alg2 = _time_and_memory(
                        lambda: algorithm2_from_Aks(
                            kraus, (interval.a, interval.b), with_timing=True
                        )
                    )

                    brute_truth, t_brute, mem_brute = _time_and_memory(
                        lambda: _eval_predicate(gamma, interval, config.N_max)
                    )
                    brute_plus, brute_minus = _tail_eventually(brute_truth, config.tail_window)

                    schur_vals = eigvals
                    schur_gamma = lambda N: float(np.sum(schur_vals ** N).real)
                    schur_truth, t_schur, mem_schur = _time_and_memory(
                        lambda: _eval_predicate(schur_gamma, interval, config.N_max)
                    )
                    schur_plus, schur_minus = _tail_eventually(schur_truth, config.tail_window)

                    agree = sum(
                        1 for a, b in zip(brute_truth, schur_truth) if a == b
                    ) / config.N_max
                    alg2_pred = [
                        _alg2_predicate(
                            alg2.omega_plus, alg2.omega_minus, alg2.info.kappa, n
                        )
                        for n in range(1, config.N_max + 1)
                    ]
                    alg2_agree = sum(
                        1 for a, b in zip(brute_truth, alg2_pred) if a == b
                    ) / config.N_max

                    results[(D, eps, seed, repeat)] = BenchmarkResult(
                        runtime={"alg2": t_alg2, "brute": t_brute, "schur": t_schur},
                        memory={
                            "alg2": mem_alg2,
                            "brute": mem_brute,
                            "schur": mem_schur,
                        },
                        metrics={
                            "tightness": float(len(brute_plus.difference(brute_minus))),
                            "agreement_schur": float(agree),
                            "agreement_alg2": float(alg2_agree),
                            "omega_plus_size": float(len(alg2.omega_plus)),
                            "omega_minus_size": float(len(alg2.omega_minus)),
                        },
                        info={
                            "num_blocks": alg2.info.num_blocks,
                            "kappa": alg2.info.kappa,
                            "rho2": alg2.info.rho2_global,
                            "interval": (interval.a, interval.b),
                            "timing": {
                                "decomp": alg2.info.timing.decomp
                                if alg2.info.timing
                                else None,
                                "cert": alg2.info.timing.cert if alg2.info.timing else None,
                                "logic": alg2.info.timing.logic if alg2.info.timing else None,
                            },
                        },
                    )
    return results

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

                    start = time.perf_counter()
                    alg2 = algorithm2_from_Aks(kraus, (interval.a, interval.b))
                    t_alg2 = time.perf_counter() - start

                    start = time.perf_counter()
                    brute_truth = _eval_predicate(gamma, interval, config.N_max)
                    brute_plus, brute_minus = _tail_eventually(brute_truth, config.tail_window)
                    t_brute = time.perf_counter() - start

                    start = time.perf_counter()
                    schur_vals = eigvals
                    schur_gamma = lambda N: float(np.sum(schur_vals ** N).real)
                    schur_truth = _eval_predicate(schur_gamma, interval, config.N_max)
                    schur_plus, schur_minus = _tail_eventually(schur_truth, config.tail_window)
                    t_schur = time.perf_counter() - start

                    agree = sum(
                        1 for a, b in zip(brute_truth, schur_truth) if a == b
                    ) / config.N_max

                    results[(D, eps, seed, repeat)] = BenchmarkResult(
                        runtime={"alg2": t_alg2, "brute": t_brute, "schur": t_schur},
                        memory={},
                        metrics={
                            "tightness": float(len(brute_plus.difference(brute_minus))),
                            "agreement": float(agree),
                        },
                        info={
                            "num_blocks": alg2.info.num_blocks,
                            "kappa": alg2.info.kappa,
                            "rho2": alg2.info.rho2_global,
                            "interval": (interval.a, interval.b),
                        },
                    )
    return results

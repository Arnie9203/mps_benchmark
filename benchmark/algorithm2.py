"""Implementation of Algorithm 2 (reference skeleton)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, List, Tuple

import numpy as np
import numpy.linalg as LA


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def lcm_list(xs: Iterable[int]) -> int:
    r = 1
    for x in xs:
        r = lcm(r, x)
    return r


def nullspace(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    U, s, Vh = LA.svd(A)
    rnk = np.sum(s > tol)
    return Vh[rnk:].conj().T


def superop_matrix(kraus: List[np.ndarray]) -> np.ndarray:
    d = kraus[0].shape[0]
    M = np.zeros((d * d, d * d), dtype=complex)
    for B in kraus:
        M += np.kron(B.conj(), B)
    return M


def spectral_radius_and_eigs(M: np.ndarray) -> Tuple[float, np.ndarray]:
    eigvals = LA.eigvals(M)
    r = float(np.max(np.abs(eigvals)))
    return r, eigvals


def peripheral_period(
    eigvals: np.ndarray,
    r: float,
    tol: float = 1e-10,
    max_den: int = 64,
) -> int:
    denoms = []
    for lam in eigvals:
        if abs(abs(lam) - r) <= tol:
            theta = np.angle(lam)
            x = (theta / (2 * math.pi)) % 1.0
            frac = Fraction(x).limit_denominator(max_den)
            denoms.append(frac.denominator)
    return lcm_list(denoms) if denoms else 1


def apply_E(Aks: List[np.ndarray], X: np.ndarray) -> np.ndarray:
    Y = np.zeros_like(X, dtype=complex)
    for A in Aks:
        Y += A @ X @ A.conj().T
    return Y


def hermitian_from_vec(v: np.ndarray, d: int) -> np.ndarray:
    Y = v.reshape((d, d), order="F")
    return (Y + Y.conj().T) / 2


def pos_neg_parts(H: np.ndarray, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    w, U = LA.eigh(H)
    wp = np.clip(w, 0, None)
    wn = np.clip(-w, 0, None)
    Hp = (U * wp) @ U.conj().T
    Hn = (U * wn) @ U.conj().T
    Hp[np.abs(Hp) < tol] = 0
    Hn[np.abs(Hn) < tol] = 0
    return Hp, Hn


def support_basis_psd(Psd: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    w, U = LA.eigh(Psd)
    idx = np.where(w > tol)[0]
    if len(idx) == 0:
        return np.zeros((Psd.shape[0], 0), dtype=complex)
    B = U[:, idx]
    Q, _ = LA.qr(B)
    return Q


def basis_union(bases: Iterable[np.ndarray], tol: float = 1e-10) -> np.ndarray:
    cols = [B for B in bases if B.shape[1] > 0]
    if not cols:
        first = next(iter(bases))
        return np.zeros((first.shape[0], 0), dtype=complex)
    M = np.concatenate(cols, axis=1)
    Q, _ = LA.qr(M)
    U, s, _ = LA.svd(Q, full_matrices=False)
    r = np.sum(s > tol)
    return U[:, :r]


def orth_complement(B: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if B.shape[1] == 0:
        return np.eye(B.shape[0], dtype=complex)
    N = nullspace(B.conj().T, tol=tol)
    Q, _ = LA.qr(N)
    return Q


def restrict_kraus(Aks: List[np.ndarray], Q: np.ndarray) -> List[np.ndarray]:
    return [Q.conj().T @ A @ Q for A in Aks]


def dedup_psd_list(G: List[np.ndarray], tol: float = 1e-8) -> List[np.ndarray]:
    out = []
    for X in G:
        tr = float(np.trace(X).real)
        if tr <= tol:
            continue
        Xn = X / tr
        keep = True
        for Y in out:
            if LA.norm(Xn - Y) < tol:
                keep = False
                break
        if keep:
            out.append(Xn)
    return out


def decompose_irreducible_by_support(
    Aks: List[np.ndarray],
    Q: np.ndarray | None = None,
    depth: int = 0,
    max_depth: int = 30,
) -> List[np.ndarray]:
    d = Aks[0].shape[0]
    if Q is None:
        Q = np.eye(d, dtype=complex)

    m = Q.shape[1]
    if m <= 1 or depth >= max_depth:
        return [Q]

    A_sub = restrict_kraus(Aks, Q)
    M = superop_matrix(A_sub)

    r, eigvals = spectral_radius_and_eigs(M)

    N = nullspace(M - r * np.eye(m * m, dtype=complex), tol=1e-8)
    if N.shape[1] == 0:
        return [Q]

    herm_solutions = []
    for j in range(N.shape[1]):
        H = hermitian_from_vec(N[:, j], m)
        if LA.norm(apply_E(A_sub, H) - r * H) < 1e-6:
            herm_solutions.append(H)
    if not herm_solutions:
        return [Q]

    Gamma = []
    for H in herm_solutions:
        Hp, Hn = pos_neg_parts(H)
        if np.trace(Hp).real > 1e-10:
            Gamma.append(Hp)
        if np.trace(Hn).real > 1e-10:
            Gamma.append(Hn)
    Gamma = dedup_psd_list(Gamma, tol=1e-6)
    if not Gamma:
        return [Q]

    supp_bases = [support_basis_psd(G, tol=1e-8) for G in Gamma]

    for B in supp_bases:
        if 0 < B.shape[1] < m:
            Q0 = Q @ B
            Q1 = Q @ orth_complement(B, tol=1e-8)
            return (
                decompose_irreducible_by_support(Aks, Q0, depth + 1, max_depth)
                + decompose_irreducible_by_support(Aks, Q1, depth + 1, max_depth)
            )

    V = basis_union(supp_bases, tol=1e-8)
    if 0 < V.shape[1] < m:
        Q0 = Q @ V
        Q1 = Q @ orth_complement(V, tol=1e-8)
        return (
            decompose_irreducible_by_support(Aks, Q0, depth + 1, max_depth)
            + decompose_irreducible_by_support(Aks, Q1, depth + 1, max_depth)
        )

    for i, Gi in enumerate(Gamma):
        for j in range(i + 1, len(Gamma)):
            X = (Gi - Gamma[j])
            X = (X + X.conj().T) / 2
            Xp, Xn = pos_neg_parts(X)
            Bp = support_basis_psd(Xp, tol=1e-8)
            Bn = support_basis_psd(Xn, tol=1e-8)
            if 0 < Bp.shape[1] < m and 0 < Bn.shape[1] < m:
                Q0 = Q @ Bp
                Q1 = Q @ Bn
                return (
                    decompose_irreducible_by_support(Aks, Q0, depth + 1, max_depth)
                    + decompose_irreducible_by_support(Aks, Q1, depth + 1, max_depth)
                )

    return [Q]


def check_invariance(Aks: List[np.ndarray], Q: np.ndarray, tol: float = 1e-8) -> Tuple[bool, float]:
    P = Q @ Q.conj().T
    I = np.eye(P.shape[0], dtype=complex)
    for A in Aks:
        v = LA.norm((I - P) @ A @ P)
        if v > tol:
            return False, v
    return True, 0.0


@dataclass
class Algorithm2Timing:
    decomp: float
    cert: float
    logic: float

    @property
    def total(self) -> float:
        return self.decomp + self.cert + self.logic


@dataclass
class Algorithm2Info:
    num_blocks: int
    block_dims: List[int]
    rm_each_block: List[float]
    pm_each_block: List[int]
    kappa: int
    rho2_global: float
    nonperiph_count: int
    timing: Algorithm2Timing | None = None


@dataclass
class Algorithm2Output:
    omega_plus: set[int]
    omega_minus: dict[int, dict[str, object]]
    info: Algorithm2Info
    gamma: callable
    blocks_Q: List[np.ndarray]
    blocks_kraus: List[List[np.ndarray]]


def algorithm2_from_Aks(
    Aks: List[np.ndarray],
    interval_I: Tuple[float, float],
    tol_periph: float = 1e-10,
    *,
    with_timing: bool = False,
) -> Algorithm2Output:
    a, b = interval_I
    t_decomp = 0.0
    t_cert = 0.0
    t_logic = 0.0
    if with_timing:
        import time
        t0 = time.perf_counter()
    blocks_Q = decompose_irreducible_by_support(Aks)
    blocks_kraus = [restrict_kraus(Aks, Qm) for Qm in blocks_Q]

    for idx, Qm in enumerate(blocks_Q):
        ok, err = check_invariance(Aks, Qm)
        if not ok:
            raise RuntimeError(f"block {idx} not invariant, err={err}")
    if with_timing:
        t1 = time.perf_counter()
        t_decomp = t1 - t0

    if with_timing:
        t2 = time.perf_counter()
    Ms, eigs_list, rms, pms = [], [], [], []
    for kraus_m in blocks_kraus:
        M = superop_matrix(kraus_m)
        Ms.append(M)
        r, eigvals = spectral_radius_and_eigs(M)
        p = peripheral_period(eigvals, r, tol=tol_periph)
        rms.append(r)
        pms.append(p)
        eigs_list.append(eigvals)

    kappa = lcm_list(pms)

    distinct_rs = sorted(set([round(x, 12) for x in rms]), reverse=True)

    periph_eigs = []
    nonperiph_mods = []
    for m, eigvals in enumerate(eigs_list):
        rm = rms[m]
        per = [lam for lam in eigvals if abs(abs(lam) - rm) <= tol_periph]
        periph_eigs.append(per)
        for lam in eigvals:
            if abs(abs(lam) - rm) > tol_periph:
                nonperiph_mods.append(abs(lam))
    if with_timing:
        t3 = time.perf_counter()
        t_cert = t3 - t2

    def Gamma(N: int) -> float:
        s = 0 + 0j
        for eigvals in eigs_list:
            s += np.sum(eigvals ** N)
        return float(s.real)

    def in_I(x: float) -> bool:
        return (x > a) and (x < b)

    Omega_plus_classes = set()
    Omega_minus = {}

    for i in range(kappa):
        chosen_r = None
        Ci = None
        for r in distinct_rs:
            c = 0 + 0j
            for m in range(len(Ms)):
                if abs(rms[m] - r) <= 1e-10:
                    for lam in periph_eigs[m]:
                        c += lam ** i
            if abs(c) > 1e-10:
                chosen_r = r
                Ci = c
                break

        if chosen_r is None:
            continue

        main_value = float(Ci.real) * (chosen_r ** i)

        if not (main_value > a and main_value < b):
            continue

        ok_dominate = True
        for m in range(len(Ms)):
            if rms[m] > chosen_r + 1e-10:
                ok_dominate = False
                break
        if not ok_dominate:
            continue

        Omega_plus_classes.add(i)

        if nonperiph_mods:
            rho2 = float(max(nonperiph_mods))
            count2 = len(nonperiph_mods)
        else:
            rho2 = 0.0
            count2 = 0

        margin = min(main_value - a, b - main_value)
        target = margin * 0.5

        def tail_bound(N: int) -> float:
            return count2 * (rho2 ** N)

        Ni = None
        t = 0
        while True:
            N = i + t * kappa
            if N <= 0:
                t += 1
                continue
            if tail_bound(N) < target:
                Ni = N
                break
            if N > 5000:
                break
            t += 1

        Lambda_i = []
        if Ni is not None:
            for N in range(i, Ni, kappa):
                if N > 0 and in_I(Gamma(N)):
                    Lambda_i.append(N)

        Omega_minus[i] = {"Ni": Ni, "exceptions": Lambda_i}

    if with_timing:
        t4 = time.perf_counter()
        t_logic = t4 - t3

    info = Algorithm2Info(
        num_blocks=len(blocks_Q),
        block_dims=[Q.shape[1] for Q in blocks_Q],
        rm_each_block=rms,
        pm_each_block=pms,
        kappa=kappa,
        rho2_global=float(max(nonperiph_mods)) if nonperiph_mods else 0.0,
        nonperiph_count=len(nonperiph_mods),
        timing=Algorithm2Timing(decomp=t_decomp, cert=t_cert, logic=t_logic)
        if with_timing
        else None,
    )

    return Algorithm2Output(
        omega_plus=Omega_plus_classes,
        omega_minus=Omega_minus,
        info=info,
        gamma=Gamma,
        blocks_Q=blocks_Q,
        blocks_kraus=blocks_kraus,
    )

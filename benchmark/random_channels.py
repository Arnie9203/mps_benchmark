"""Random channel generators with controllable block structure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import numpy.linalg as LA


def _random_unitary(rng: np.random.Generator, d: int) -> np.ndarray:
    X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = LA.qr(X)
    diag = np.diag(R)
    phases = diag / np.abs(diag)
    return Q * phases


def _normalize_kraus(kraus: List[np.ndarray]) -> List[np.ndarray]:
    d = kraus[0].shape[0]
    acc = np.zeros((d, d), dtype=complex)
    for A in kraus:
        acc += A.conj().T @ A
    w, U = LA.eigh(acc)
    w = np.clip(w, 1e-12, None)
    inv = (U * (1.0 / np.sqrt(w))) @ U.conj().T
    return [A @ inv for A in kraus]


@dataclass
class BlockSpec:
    dim: int
    period: int


@dataclass
class ChannelFamily:
    D: int
    d: int
    blocks: List[BlockSpec]
    kraus: List[np.ndarray]


def build_block_channel(
    rng: np.random.Generator,
    D: int,
    d: int = 3,
    periods: Sequence[int] | None = None,
) -> ChannelFamily:
    if periods is None:
        periods = [1, 2]
    num_blocks = max(1, len(periods))
    base = D // num_blocks
    dims = [base] * num_blocks
    dims[-1] += D - sum(dims)

    blocks = []
    block_kraus = []
    for dim, period in zip(dims, periods):
        blocks.append(BlockSpec(dim=dim, period=period))
        U = _random_unitary(rng, dim)
        local = []
        for _ in range(d):
            A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
            A = A / np.sqrt(dim)
            local.append(U @ A @ U.conj().T)
        local = _normalize_kraus(local)
        if period == 2:
            phase = np.diag([1 if i % 2 == 0 else -1 for i in range(dim)])
            local = [phase @ A for A in local]
        block_kraus.append(local)

    kraus = []
    for k in range(d):
        big = np.zeros((D, D), dtype=complex)
        offset = 0
        for blk, dim in zip(block_kraus, dims):
            big[offset : offset + dim, offset : offset + dim] = blk[k]
            offset += dim
        kraus.append(big)

    return ChannelFamily(D=D, d=d, blocks=blocks, kraus=kraus)


def mix_with_identity(kraus: List[np.ndarray], epsilon: float) -> List[np.ndarray]:
    d = kraus[0].shape[0]
    identity = np.eye(d, dtype=complex)
    scaled = [np.sqrt(1 - epsilon) * A for A in kraus]
    scaled.append(np.sqrt(epsilon) * identity)
    return scaled

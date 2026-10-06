from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

RiskSampler = Callable[[str, str], float]
PromptBuilder = Callable[[Sequence[str]], tuple[str, str]]


@dataclass(frozen=True)
class DBSAChunkResult:
    chunk: str
    energy_distance: float
    p_value: float
    full_samples: tuple[float, ...]
    perturbed_samples: tuple[float, ...]


@dataclass(frozen=True)
class DBSAResult:
    ranked_chunks: tuple[str, ...]
    chunks: dict[str, DBSAChunkResult]


def energy_distance_1d(
    x: Sequence[float],
    y: Sequence[float],
) -> float:
    """
    Scalar adaptation of DBSA's compute_energy_distance().

    The original DBSA implementation computes pairwise distances between
    response embeddings. Here each LLM response is represented by one
    scalar mortality probability, so the pairwise metric is |x - y|.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError(
            "energy_distance_1d requires non-empty samples"
        )

    n = len(x_arr)
    m = len(y_arr)

    dists_xy = np.abs(
        x_arr[:, None] - y_arr[None, :]
    )
    dists_xx = np.abs(
        x_arr[:, None] - x_arr[None, :]
    )
    dists_yy = np.abs(
        y_arr[:, None] - y_arr[None, :]
    )

    term1 = (2.0 / (n * m)) * np.sum(dists_xy)
    term2 = (1.0 / n**2) * np.sum(dists_xx)
    term3 = (1.0 / m**2) * np.sum(dists_yy)

    return float(term1 - term2 - term3)

def permutation_test_energy_1d(
    x: Sequence[float],
    y: Sequence[float],
    *,
    num_permutations: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Scalar adaptation of DBSA's permutation_test_energy().
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    combined = np.concatenate((x_arr, y_arr))
    n = len(x_arr)

    rng = np.random.default_rng(seed)

    energy_values = []

    for _ in range(num_permutations):
        shuffled = rng.permutation(combined)

        perm_x = shuffled[:n]
        perm_y = shuffled[n:]

        e_perm = energy_distance_1d(
            perm_x,
            perm_y,
        )

        energy_values.append(e_perm)

    return np.asarray(energy_values)

def compute_energy_distance_fn_1d(
    x: Sequence[float],
    y: Sequence[float],
    *,
    num_permutations: int = 500,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Scalar adaptation of DBSA's compute_energy_distance_fn().
    """
    observed = energy_distance_1d(
        x,
        y,
    )

    permutation_values = permutation_test_energy_1d(
        x,
        y,
        num_permutations=num_permutations,
        seed=seed,
    )

    p_value = np.mean(
        permutation_values >= observed
    )

    return observed, float(p_value)


def sample_risk_distribution(
    *,
    sampler: RiskSampler,
    system_prompt: str,
    user_prompt: str,
    n_samples: int,
    max_workers: int = 1,
) -> list[float]:
    """
    Draw repeated stochastic mortality-risk samples from the LLM.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    def sample_once(_: int) -> float:
        value = float(
            sampler(
                system_prompt,
                user_prompt,
            )
        )

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Risk sample outside [0, 1]: {value}")

        return value

    if max_workers == 1:
        return [sample_once(i) for i in range(n_samples)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        values = list(
            executor.map(
                sample_once,
                range(n_samples),
            )
        )

    return values


def dbsa_risk_rank(
    *,
    chunk_names: Sequence[str],
    prompt_builder: PromptBuilder,
    sampler: RiskSampler,
    n_full: int = 3,
    n_perturbed: int = 3,
    n_permutations: int = 100,
    max_workers: int = 1,
    seed: int = 0,
) -> DBSAResult:
    """
    Rank clinical chunks by leave-one-chunk-out DBSA risk sensitivity.

    The full patient prompt is sampled repeatedly to obtain a baseline
    mortality-risk distribution. Each clinical chunk is then removed in
    turn and another risk distribution is sampled.

    Chunk importance is the energy distance between those distributions.
    """
    chunks = tuple(chunk_names)

    if not chunks:
        raise ValueError("chunk_names must not be empty")

    system_prompt, full_user_prompt = prompt_builder(chunks)

    full_samples = sample_risk_distribution(
        sampler=sampler,
        system_prompt=system_prompt,
        user_prompt=full_user_prompt,
        n_samples=n_full,
        max_workers=max_workers,
    )

    results: dict[str, DBSAChunkResult] = {}

    for index, chunk in enumerate(chunks):
        retained = tuple(candidate for candidate in chunks if candidate != chunk)

        (
            perturbed_system_prompt,
            perturbed_user_prompt,
        ) = prompt_builder(retained)

        perturbed_samples = sample_risk_distribution(
            sampler=sampler,
            system_prompt=perturbed_system_prompt,
            user_prompt=perturbed_user_prompt,
            n_samples=n_perturbed,
            max_workers=max_workers,
        )

        energy_distance, p_value = (
            compute_energy_distance_fn_1d(
                full_samples,
                perturbed_samples,
                num_permutations=n_permutations,
                seed=seed + index,
            )
        )

        results[chunk] = DBSAChunkResult(
            chunk=chunk,
            energy_distance=energy_distance,
            p_value=p_value,
            full_samples=tuple(full_samples),
            perturbed_samples=tuple(perturbed_samples),
        )

    ranked = tuple(
        sorted(
            chunks,
            key=lambda chunk: results[chunk].energy_distance,
            reverse=True,
        )
    )

    return DBSAResult(
        ranked_chunks=ranked,
        chunks=results,
    )

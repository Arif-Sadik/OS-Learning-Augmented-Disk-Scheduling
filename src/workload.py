"""Synthetic disk request workload generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


WorkloadType = Literal["uniform", "localized", "directional", "bursty"]


@dataclass(frozen=True)
class WorkloadSample:
    workload_type: str
    requests: list[int]
    initial_head: int
    seed: int
    phase: str | None = None


def _clip(values: np.ndarray, min_cylinder: int, max_cylinder: int) -> list[int]:
    return np.clip(np.rint(values), min_cylinder, max_cylinder).astype(int).tolist()


def generate_uniform(
    rng: np.random.Generator,
    count: int,
    min_cylinder: int = 0,
    max_cylinder: int = 199,
) -> list[int]:
    """Generate requests broadly distributed across the full disk."""

    return rng.integers(min_cylinder, max_cylinder + 1, size=count).astype(int).tolist()


def generate_localized(
    rng: np.random.Generator,
    count: int,
    min_cylinder: int = 0,
    max_cylinder: int = 199,
) -> list[int]:
    """Generate requests clustered around one disk region."""

    center = int(rng.integers(min_cylinder + 25, max_cylinder - 24))
    spread = rng.uniform(6, 16)
    values = rng.normal(center, spread, size=count)
    return _clip(values, min_cylinder, max_cylinder)


def generate_directional(
    rng: np.random.Generator,
    count: int,
    min_cylinder: int = 0,
    max_cylinder: int = 199,
) -> list[int]:
    """Generate mostly upward sequential requests with small jitter."""

    start_low = int(rng.integers(min_cylinder, min_cylinder + 40))
    start_high = int(rng.integers(max_cylinder - 39, max_cylinder + 1))
    base = np.linspace(start_low, start_high, count)
    jitter = rng.normal(0, 2, size=count)
    return _clip(base + jitter, min_cylinder, max_cylinder)


def generate_bursty(
    rng: np.random.Generator,
    count: int,
    min_cylinder: int = 0,
    max_cylinder: int = 199,
) -> list[int]:
    """Generate locality bursts that jump between disk regions."""

    regions = [
        int(rng.integers(min_cylinder + 10, min_cylinder + 60)),
        int(rng.integers(min_cylinder + 70, max_cylinder - 69)),
        int(rng.integers(max_cylinder - 59, max_cylinder - 9)),
    ]
    rng.shuffle(regions)
    chunks = np.array_split(np.arange(count), 3)
    requests: list[int] = []
    for chunk, center in zip(chunks, regions):
        values = rng.normal(center, rng.uniform(5, 13), size=len(chunk))
        requests.extend(_clip(values, min_cylinder, max_cylinder))
    return requests


def generate_workload(
    workload_type: WorkloadType,
    seed: int,
    count: int = 40,
    initial_head: int = 50,
    min_cylinder: int = 0,
    max_cylinder: int = 199,
    phase: str | None = None,
) -> WorkloadSample:
    """Generate one deterministic workload sample."""

    if count < 0:
        raise ValueError("count must be non-negative")
    if not min_cylinder <= initial_head <= max_cylinder:
        raise ValueError("initial_head is outside the disk cylinder range")

    rng = np.random.default_rng(seed)
    generators = {
        "uniform": generate_uniform,
        "localized": generate_localized,
        "directional": generate_directional,
        "bursty": generate_bursty,
    }
    if workload_type not in generators:
        raise ValueError(f"unknown workload type: {workload_type}")

    requests = generators[workload_type](rng, count, min_cylinder, max_cylinder)
    return WorkloadSample(workload_type, requests, initial_head, seed, phase)


def generate_workload_shift(
    seed: int = 9000,
    windows_per_phase: int = 10,
    count: int = 40,
    initial_head: int = 50,
    min_cylinder: int = 0,
    max_cylinder: int = 199,
) -> list[WorkloadSample]:
    """Generate a reproducible shift from directional to random access."""

    samples: list[WorkloadSample] = []
    for index in range(windows_per_phase):
        samples.append(
            generate_workload(
                "directional",
                seed + index,
                count,
                initial_head,
                min_cylinder,
                max_cylinder,
                phase="phase_1_directional",
            )
        )
    for index in range(windows_per_phase):
        samples.append(
            generate_workload(
                "uniform",
                seed + 1000 + index,
                count,
                initial_head,
                min_cylinder,
                max_cylinder,
                phase="phase_2_uniform",
            )
        )
    return samples

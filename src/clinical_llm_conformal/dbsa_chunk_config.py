"""Validated config-defined partitions for clinical DBSA perturbation."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ALLOWED_DOMAINS = frozenset({"demographics", "diagnoses", "medications"})


def load_chunk_set(
    path: str | Path,
    name: str,
    *,
    available_features: list[str],
) -> dict[str, dict[str, list[str]]]:
    """Load a disjoint, exhaustive partition of features and supported domains.

    Each chunk has optional `features` and `domains` lists. All available
    measurements and all three non-measurement domains must appear once.
    """
    payload: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("chunk_sets"), dict):
        raise ValueError("Expected top-level YAML mapping 'chunk_sets'")
    if name not in payload["chunk_sets"]:
        raise ValueError(f"Unknown chunk set {name!r}; available: {list(payload['chunk_sets'])}")
    definition = payload["chunk_sets"][name]
    chunks = definition.get("chunks") if isinstance(definition, dict) else None
    if not isinstance(chunks, dict) or not chunks:
        raise ValueError(f"{name}: chunks must be a nonempty mapping")

    normalised: dict[str, dict[str, list[str]]] = {}
    all_features: list[str] = []
    all_domains: list[str] = []
    for chunk_name, spec in chunks.items():
        if not isinstance(chunk_name, str) or not chunk_name.strip():
            raise ValueError("Chunk names must be nonblank strings")
        if not isinstance(spec, dict) or set(spec) - {"features", "domains"}:
            raise ValueError(f"{chunk_name}: only features/domains lists are supported")
        feats = spec.get("features", [])
        domains = spec.get("domains", [])
        if not isinstance(feats, list) or not all(isinstance(x, str) for x in feats):
            raise ValueError(f"{chunk_name}: features must be a list of strings")
        if not isinstance(domains, list) or not all(isinstance(x, str) for x in domains):
            raise ValueError(f"{chunk_name}: domains must be a list of strings")
        if not feats and not domains:
            raise ValueError(f"{chunk_name}: chunk must contain a feature or domain")
        normalised[chunk_name] = {"features": feats, "domains": domains}
        all_features.extend(feats)
        all_domains.extend(domains)

    expected = set(available_features)
    if len(expected) != len(available_features):
        raise ValueError("Input measurement features contain duplicate names")
    counts = Counter(all_features)
    missing = expected - set(counts)
    extras = set(counts) - expected
    repeated = sorted(feature for feature, n in counts.items() if n > 1)
    if missing or extras or repeated:
        raise ValueError(
            f"{name}: feature partition invalid: missing={sorted(missing)}, "
            f"unknown={sorted(extras)}, duplicate={repeated}"
        )

    domain_counts = Counter(all_domains)
    missing_domains = ALLOWED_DOMAINS - set(domain_counts)
    extra_domains = set(domain_counts) - ALLOWED_DOMAINS
    duplicate_domains = sorted(d for d, n in domain_counts.items() if n > 1)
    if missing_domains or extra_domains or duplicate_domains:
        raise ValueError(
            f"{name}: domain partition invalid: missing={sorted(missing_domains)}, "
            f"unknown={sorted(extra_domains)}, duplicate={duplicate_domains}"
        )
    return normalised

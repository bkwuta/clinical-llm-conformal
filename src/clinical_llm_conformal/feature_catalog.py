"""Metadata for features supplied by source-independent measurement preprocessors.

Catalog data are presentation metadata, not evidence that any feature was
observed at a valid prediction cutoff or that reference limits are clinical rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    display_name: str
    kind: str
    unit: str | None = None
    reference_range: str | None = None


def load_feature_catalog(path: str | Path) -> dict[str, FeatureSpec]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("features"), dict):
        raise ValueError("Feature catalog must be a YAML mapping with a 'features' mapping")
    if not document["features"]:
        raise ValueError("Feature catalog cannot be empty")

    catalog: dict[str, FeatureSpec] = {}
    for name, raw in document["features"].items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Feature keys must be nonempty strings")
        if not isinstance(raw, dict):
            raise ValueError(f"Feature {name!r} must be a mapping")
        unknown = set(raw) - {"display_name", "type", "unit", "reference_range"}
        if unknown:
            raise ValueError(f"Feature {name!r}: unsupported metadata keys {sorted(unknown)}")
        kind = raw.get("type")
        if kind not in {"numeric", "categorical"}:
            raise ValueError(f"Feature {name!r}: type must be numeric or categorical")
        display = raw.get("display_name", name)
        if not isinstance(display, str) or not display.strip():
            raise ValueError(f"Feature {name!r}: invalid display_name")
        for field in ("unit", "reference_range"):
            value = raw.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Feature {name!r}: {field} must be text or null")
        catalog[name] = FeatureSpec(
            name=name,
            display_name=display,
            kind=kind,
            unit=raw.get("unit"),
            reference_range=raw.get("reference_range"),
        )
    return catalog

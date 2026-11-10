"""Synthetic Stage 6E end-to-end smoke test; requires no MIMIC or LLM server."""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="clinical_stage6e_") as tmp:
        root = Path(tmp)
        cohort = root / "cohort.parquet"
        measurements = root / "measurements.parquet"
        config = root / "config.yaml"
        output = root / "results.csv"
        pd.DataFrame([
            {"subject_id": 1001, "hadm_id": 2001, "label": 1},
            {"subject_id": 1002, "hadm_id": 2002, "label": 0},
        ]).to_parquet(cohort, index=False)
        pd.DataFrame([
            (1001, 2001, "2133-09-18 08:00:00", "Heart Rate", "84"),
            (1001, 2001, "2133-09-18 08:00:00", "Custom Biomarker", "1.2"),
        ], columns=["subject_id", "hadm_id", "timestamp", "feature", "value"]).to_parquet(measurements, index=False)
        config.write_text(yaml.safe_dump({
            "cohort": {"path": str(cohort)},
            "measurements": {"provider": "generic", "path": str(measurements)},
            "features": {"catalog": "configs/feature_catalog_example.yaml", "selected": ["Heart Rate", "Custom Biomarker"]},
            "prompt": {"provider": "generic", "task": "mortality", "n_shot": 0},
        }), encoding="utf-8")
        subprocess.run([
            sys.executable, "scripts/run_configured_clinical.py", "--config", str(config), "--output", str(output)
        ], check=True)
        with output.open(newline="", encoding="utf-8") as stream:
            results = list(csv.DictReader(stream))
        assert len(results) == 2, results
        by_id = {int(r["hadm_id"]): r for r in results}
        assert by_id[2001]["status"] == "prompt_validated", by_id
        assert by_id[2001]["prompt_sha256"], by_id
        assert by_id[2002]["status"] == "no_measurements", by_id
        assert by_id[2002]["label"] == "0", by_id
        assert all(not r["model"] for r in results), results
        print("Stage 6E configurable runner: PASS")


if __name__ == "__main__":
    main()

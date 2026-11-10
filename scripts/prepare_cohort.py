"""Generate a portable MIMIC hospital-admission cohort manifest.

Examples:
 uv run python scripts/prepare_cohort.py --provider clinicrealm \\
   --source tmp/clinicrealm_reference/processed/clinicrealm_structured_test_200.parquet \\
   --output /data/mimiciv/cohorts/clinicrealm_200.parquet
 uv run python scripts/prepare_cohort.py --provider file \\
   --source /data/mimiciv/cohorts/custom_admissions.csv \\
   --sample 500 --seed 42 --output /data/mimiciv/cohorts/custom_500.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

from clinical_llm_conformal.cohort import load_cohort, sample_cohort, save_cohort
from clinical_llm_conformal.cohort_adapters import cohort_from_clinicrealm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("clinicrealm", "file"), required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of admissions to sample (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mimic-version", default="unknown",
                        help="Provenance only; provide source version when known")
    args = parser.parse_args()

    if args.provider == "clinicrealm":
        frame = cohort_from_clinicrealm(args.source)
    else:
        frame = load_cohort(args.source)
    total = len(frame)
    frame = sample_cohort(frame, n_admissions=args.sample, seed=args.seed)
    path = save_cohort(frame, args.output, provenance={
        "provider": args.provider,
        "source": str(args.source),
        "mimic_version": args.mimic_version,
        "sampling_seed": args.seed if args.sample is not None else None,
        "original_n_admissions": total,
        "prediction_time_status": (
            "not_supplied_by_clinicrealm_adapter" if args.provider == "clinicrealm"
            else "as_provided_by_input_if_present"
        ),
    })
    print(f"Cohort saved: {path}")
    print(f"Admissions: {len(frame)} | Unique patients: {frame.subject_id.nunique()}")
    print(f"Columns: {list(frame.columns)}")
    print(f"Metadata: {path.with_suffix('.metadata.json')}")


if __name__ == "__main__":
    main()

"""Adapters from upstream preprocessing outputs to admission-level manifests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cohort import validate_cohort
from .data import LABTEST_FEATURES, build_patient_records, load_clinicrealm_ehr


def cohort_from_clinicrealm(path: str | Path) -> pd.DataFrame:
    """Preserve the original ClinicRealm admission cohort and outcome labels.

    ClinicRealm's longitudinal dates do not automatically define the time at
    which outcome prediction would have been performed. Therefore this adapter
    intentionally leaves prediction_time and stay_id undefined.
    """
    ehr = load_clinicrealm_ehr(Path(path))
    records = build_patient_records(ehr, features=LABTEST_FEATURES)
    rows = [
        {"subject_id": int(patient_id),
         "hadm_id": int(admission_id),
         "label": int(record["label"])}
        for (patient_id, admission_id), record in records.items()
    ]
    return validate_cohort(pd.DataFrame(rows))

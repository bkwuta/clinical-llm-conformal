# Clinical LLM Conformal

A reproducible research pipeline for evaluating local large language models on structured MIMIC-IV EHR data, analyzing prediction sensitivity with a clinical adaptation of DBSA, and extending the pipeline with conformal-validity methods.

## Current pipeline

```text
Local MIMIC-IV structured data
        ↓
Pinned PKU-AICare MIMIC-IV preprocessing
        ↓
Structured-only ClinicRealm-style 200-admission test split
        ↓
ClinicRealm prompt construction
        ↓
Local OpenAI-compatible LLM
        ↓
Mortality probability parsing
        ↓
Clinical-chunk DBSA
        ↓
Later: sufficient-subset + conformal analysis
```

The current frozen structured-only test cohort contains:

| Quantity | Value |
|---|---:|
| Admissions | 200 |
| Unique patients | 199 |
| Alive | 177 |
| Dead | 23 |
| Longitudinal rows / visits | 773 |

This is **not the exact published ClinicRealm MIMIC-IV test cohort**. ClinicRealm's benchmark cohort was restricted to admissions with available discharge notes before the fixed test split. This project currently does not have MIMIC-IV-Note access, so the note-availability restriction is omitted.

The cohort produced here should therefore be described as a:

> **structured-only ClinicRealm-style MIMIC-IV test split**

rather than as an exact reproduction of ClinicRealm's published 200-case test set.

## Repository philosophy

The repository separates three concerns:

```text
Data acquisition       ← outside this repository
Research preprocessing ← this repository
LLM / DBSA / conformal ← this repository
```

Users may obtain MIMIC-IV data through PhysioNet, an institutional data store, BigQuery, or another authorized source. The repository only requires the files to be placed in the documented directory structure.

Google Cloud / BigQuery credentials, billing-project configuration, and BigQuery-specific acquisition code are intentionally **not part of the public workflow**.

## Required data layout

Set the root directory in `.env`:

```env
MIMIC_DATA_DIR=/path/to/mimiciv
```

The directory must contain:

```text
${MIMIC_DATA_DIR}/
├── hosp/
│   ├── patients.csv
│   └── admissions.csv
└── icu/
    ├── icustays.csv
    └── chartevents.csv
```

The current structured-EHR pipeline does not require discharge-note text.

MIMIC data must never be committed to this repository.

## Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Example:

```env
MIMIC_DATA_DIR=/path/to/mimiciv

LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=qwen3.8-27b-local
LLM_API_KEY=local
```

Install with:

```bash
uv sync
```

## Pinned upstream dependencies

The project vendors the upstream preprocessing / benchmark repositories as pinned submodules:

```text
vendor/
├── mimic_preprocessor/
└── ehr-llm-benchmark/
```

Do not modify vendored source code. Project-specific adaptations belong in this repository's wrapper layer.

## Preparing the structured reference cohort

The central configuration is:

```text
configs/clinicrealm_reference.yaml
```

Recommended structure:

```yaml
seed: 42

data:
  work_root: tmp/clinicrealm_reference
  test_ehr_path: tmp/clinicrealm_reference/processed/clinicrealm_structured_test_200.parquet

reference:
  test_size: 200
```

Run:

```bash
uv run python scripts/prepare_clinicrealm_reference.py
```

The script:

1. reads `MIMIC_DATA_DIR` from `.env`;
2. validates the four required MIMIC-IV files;
3. invokes the pinned PKU-AICare MIMIC-IV preprocessing code;
4. creates `RecordID = PatientID_AdmissionID`;
5. applies the ClinicRealm-style stratified split using `seed=42` and `test_size=200`;
6. writes the frozen structured-only test cohort.

Expected output artifact:

```text
tmp/clinicrealm_reference/processed/
└── clinicrealm_structured_test_200.parquet
```

The current validated cohort is:

```text
rows:       773
admissions: 200
patients:   199
Outcome 0:  177
Outcome 1:   23
```

### Why the numbers differ from ClinicRealm

The published ClinicRealm MIMIC-IV structured-EHR test set used a parent cohort restricted by discharge-note availability. Its reported 200-case test split therefore has a different mortality balance and number of longitudinal rows.

This repository currently lacks that discharge-note eligibility information. It intentionally does **not** force the resulting split to match the published class counts.

If MIMIC-IV-Note admission identifiers become available later, the existing `restrict_to_clinicrealm_cohort(...)` helper can be used before splitting to restore the note-restricted path.

## Prompt construction smoke test

Prompt construction uses the frozen structured test cohort configured in `configs/clinicrealm_reference.yaml`.

Run:

```bash
uv run python scripts/test_prompt_construction.py
```

This checks structured EHR loading, patient/admission record construction, ClinicRealm prompt assets, longitudinal record formatting, feature names and units, and final system/user prompts.

## Single-prompt LLM smoke test

Start the local OpenAI-compatible model server, then run:

```bash
uv run python scripts/run_single_prompt.py
```

The smoke test selects the admission with the richest longitudinal record in the configured cohort and sends a zero-shot mortality prompt to the local model.

The response is parsed into a mortality probability.

## DBSA

DBSA is currently used as a **clinical-chunk sensitivity analysis** over structured EHR prompt components.

Clinical chunks include:

- Demographics
- Hemodynamics (HR/BP)
- Respiratory status
- Temperature
- Neuro status (GCS)
- Capillary refill
- Glucose
- Blood pH
- Weight
- Height

DBSA parameters live in:

```text
configs/dbsa.yaml
```

Example:

```yaml
seed: 42

prompt:
  task: mortality
  n_shot: 0
  unit: true
  reference_range: true
  format_type: list

generation:
  temperature: 0.8
  max_tokens: 128
  enable_thinking: false
  reasoning_effort: none
  top_k: 20
  min_p: 0.0

dbsa:
  n_full: 4
  n_perturbed: 4
  n_permutations: 500
  max_workers: 1
```

The EHR cohort path is **not duplicated in `dbsa.yaml`**. `run_single_dbsa.py` reads the cohort path from `configs/clinicrealm_reference.yaml`.

Run:

```bash
uv run python scripts/run_single_dbsa.py
```

### How the single-patient DBSA smoke test selects a case

If `configs/dbsa.yaml` contains a `patient:` block, that exact patient/admission is used.

It should **not** be interpreted as representative patient sampling for the final study.

For a formal DBSA analysis, the intended next step is to define a fixed, reproducible evaluation sample from the 200 admissions, preferably stratified by outcome and selected with a fixed seed.

## DBSA interpretation

For each clinical chunk, DBSA compares mortality-probability samples from:

```text
full prompt
vs.
prompt with that chunk removed
```

and reports an energy-distance effect plus a permutation p-value.

The current implementation is a clinical adaptation of DBSA rather than an exact token-level reproduction. Because smoke tests may use only a small number of model samples, permutation p-values should not be over-interpreted.

## Low-memory preprocessing

The public pipeline uses the normal pinned upstream preprocessing path.

Some machines may not have enough RAM for the upstream global event pivot on the full MIMIC-IV cohort. That is an environment-specific execution issue and is deliberately **not built into the public research pipeline**.

```text
tmp/clinicrealm_reference/processed/mimic4_formatted_ehr.parquet
```

After that artifact exists, the normal public preparation script automatically reuses it.

This workaround changes execution strategy, not the intended preprocessing semantics. It should be validated against the ordinary processor on a subset that fits in memory.

## Development benchmark cohort

`tmp/benchmark_cohort` is a development/smoke-test workspace and is distinct from the frozen structured-only reference cohort.

Do not use an old development cohort as the default input for DBSA or reference LLM evaluation.

The canonical evaluation input is configured through:

```yaml
data:
  test_ehr_path: tmp/clinicrealm_reference/processed/clinicrealm_structured_test_200.parquet
```

in `configs/clinicrealm_reference.yaml`.

## Reproducibility

Before running larger experiments, record hashes for the processed EHR and frozen test cohort:

```bash
sha256sum   tmp/clinicrealm_reference/processed/mimic4_formatted_ehr.parquet   tmp/clinicrealm_reference/processed/clinicrealm_structured_test_200.parquet
```

The frozen cohort should satisfy:

```text
773 rows
200 unique RecordID values
199 unique PatientID values
177 Outcome=0 admissions
23 Outcome=1 admissions
```

## Data security

MIMIC-derived data and generated patient-level artifacts must remain local and must not be committed.

Recommended `.gitignore` coverage:

```gitignore
.env

data/
datasets/
cache/
artifacts/
outputs/
runs/
tmp/

*.parquet
```

Raw MIMIC CSVs should remain outside the repository through `MIMIC_DATA_DIR`.

## Current research roadmap

```text
1. Freeze structured-only ClinicRealm-style cohort
2. Verify prompt construction
3. Verify local LLM inference + probability parsing
4. Verify clinical-chunk DBSA
5. Define reproducible multi-patient DBSA evaluation sample
6. Develop sufficient-subset analysis
7. Add conformal-validity layer
8. Evaluate marginal and subgroup / conditional behavior
```

The DBSA and conformal components should remain conceptually separate:

- **DBSA** asks which clinical information materially changes the model output.
- **Conformal methods** address uncertainty / validity guarantees.

## Important methodological note

Do not claim exact reproduction of the published ClinicRealm MIMIC-IV test cohort unless the discharge-note availability restriction is restored.

Preferred terminology:

> structured-only ClinicRealm-style MIMIC-IV test split

Avoid:

> exact ClinicRealm MIMIC-IV test set

until note-restricted cohort reconstruction is available.

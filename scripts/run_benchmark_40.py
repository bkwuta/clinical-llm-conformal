from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, roc_auc_score

from clinical_llm_conformal.data import (
    LABTEST_FEATURES,
    build_patient_records,
    load_clinicrealm_ehr,
)
from clinical_llm_conformal.llm import OpenAICompatibleLLM
from clinical_llm_conformal.parsing import parse_llm_response
from clinical_llm_conformal.prompt import build_clinicrealm_prompt

CONFIG_PATH = Path("configs/benchmark.yaml")


def bootstrap_binary_metrics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)

    aurocs: list[float] = []
    auprcs: list[float] = []

    attempts = 0
    max_attempts = n_boot * 100

    while len(aurocs) < n_boot and attempts < max_attempts:
        attempts += 1

        idx = rng.integers(
            0,
            len(y),
            size=len(y),
        )

        yb = y[idx]
        pb = p[idx]

        # AUROC is undefined if a bootstrap sample has only one class.
        if len(np.unique(yb)) < 2:
            continue

        aurocs.append(float(roc_auc_score(yb, pb)))
        auprcs.append(float(average_precision_score(yb, pb)))

    if len(aurocs) != n_boot:
        raise RuntimeError(
            "Could not obtain the requested number of valid bootstrap replicates."
        )

    def summarize(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)

        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "lo95": float(np.percentile(arr, 2.5)),
            "hi95": float(np.percentile(arr, 97.5)),
        }

    return {
        "auroc": summarize(aurocs),
        "auprc": summarize(auprcs),
    }


def main() -> None:
    load_dotenv()

    config = yaml.safe_load(CONFIG_PATH.read_text())

    seed = int(config["seed"])
    benchmark = config["benchmark"]
    generation = config["generation"]

    work_root = Path(config["data"]["work_root"])
    ehr_path = work_root / "processed" / "mimic4_formatted_ehr.parquet"

    n_samples = int(benchmark["n_samples"])
    n_positive = int(benchmark["n_positive"])
    n_negative = int(benchmark["n_negative"])
    n_boot = int(benchmark["bootstrap_replicates"])

    if n_positive + n_negative != n_samples:
        raise ValueError(
            "benchmark.n_positive + benchmark.n_negative must equal "
            "benchmark.n_samples"
        )

    ehr = load_clinicrealm_ehr(ehr_path)
    patients = build_patient_records(ehr)

    positive_keys = [
        key
        for key, patient in patients.items()
        if patient["label"] == 1
    ]

    negative_keys = [
        key
        for key, patient in patients.items()
        if patient["label"] == 0
    ]

    rng = np.random.default_rng(seed)
    rng.shuffle(positive_keys)
    rng.shuffle(negative_keys)

    if len(positive_keys) < n_positive:
        raise RuntimeError(
            f"Requested {n_positive} positives, "
            f"but only {len(positive_keys)} are available."
        )

    if len(negative_keys) < n_negative:
        raise RuntimeError(
            f"Requested {n_negative} negatives, "
            f"but only {len(negative_keys)} are available."
        )

    selected = (
        positive_keys[:n_positive]
        + negative_keys[:n_negative]
    )
    rng.shuffle(selected)

    selected_labels = [
        patients[key]["label"]
        for key in selected
    ]

    unique, counts = np.unique(
        selected_labels,
        return_counts=True,
    )

    print(
        "Selected outcome counts:",
        {
            int(k): int(v)
            for k, v in zip(unique, counts, strict=True)
        },
    )

    llm = OpenAICompatibleLLM(
        model=os.environ["LLM_MODEL"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ.get("LLM_API_KEY", "local"),
    )

    def predict_one(key: tuple[int, int]) -> dict[str, object]:
        patient = patients[key]

        system_prompt, user_prompt = build_clinicrealm_prompt(
            patient_row=patient["row"],
            day_cols=patient["day_cols"],
            record_times=patient["record_times"],
            features=LABTEST_FEATURES,
            task=benchmark["task"],
            n_shot=int(benchmark["n_shot"]),
            unit=bool(benchmark["unit"]),
            reference_range=bool(
                benchmark["reference_range"]
            ),
            format_type=benchmark["format_type"],
        )

        response = llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(
                generation["temperature"]
            ),
            max_tokens=int(
                generation["max_tokens"]
            ),
            enable_thinking=bool(
                generation["enable_thinking"]
            ),
            reasoning_effort=generation[
                "reasoning_effort"
            ],
            top_k=int(generation["top_k"]),
            min_p=float(generation["min_p"]),
        )

        parsed = parse_llm_response(response.text)

        probability = (
            0.5
            if parsed.failed or parsed.probability is None
            else float(parsed.probability)
        )

        return {
            "PatientID": key[0],
            "AdmissionID": key[1],
            "Outcome": int(patient["label"]),
            "Probability": probability,
            "Failed": bool(parsed.failed),
            "PromptTokens": response.prompt_tokens,
            "CompletionTokens": response.completion_tokens,
        }

    rows: list[dict[str, object]] = []

    max_workers = int(generation["max_workers"])

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as pool:
        futures = {
            pool.submit(predict_one, key): key
            for key in selected
        }

        for index, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            key = futures[future]

            try:
                row = future.result()
            except Exception as exc:
                print(
                    f"[{index:02d}/{n_samples}] "
                    f"{key} ERROR: {exc}"
                )
                raise

            rows.append(row)

            print(
                f"[{index:02d}/{n_samples}] "
                f"({row['PatientID']}, "
                f"{row['AdmissionID']}) "
                f"y={row['Outcome']} "
                f"p={row['Probability']} "
                f"failed={row['Failed']} "
                f"completion_tokens="
                f"{row['CompletionTokens']}"
            )

    # Restore deterministic selected-admission order in the output CSV.
    order = {
        key: i
        for i, key in enumerate(selected)
    }

    rows.sort(
        key=lambda row: order[
            (
                int(row["PatientID"]),
                int(row["AdmissionID"]),
            )
        ]
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    prediction_path = (
        output_dir
        / "benchmark_40_predictions.csv"
    )

    with prediction_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    y = np.asarray(
        [row["Outcome"] for row in rows],
        dtype=int,
    )

    p = np.asarray(
        [row["Probability"] for row in rows],
        dtype=float,
    )

    point_auroc = float(
        roc_auc_score(y, p)
    )
    point_auprc = float(
        average_precision_score(y, p)
    )

    stats = bootstrap_binary_metrics(
        y,
        p,
        n_boot=n_boot,
        seed=seed,
    )

    failures = sum(
        bool(row["Failed"])
        for row in rows
    )

    print("\n===== BENCHMARK RESULT =====")
    print("N:", len(rows))
    print(
        "Failures:",
        failures,
        f"({failures / len(rows):.1%})",
    )

    print(
        f"Point AUROC: {point_auroc:.4f}"
    )
    print(
        f"Point AUPRC: {point_auprc:.4f}"
    )

    print("\nBootstrap AUROC:")
    print(stats["auroc"])

    print("\nBootstrap AUPRC:")
    print(stats["auprc"])

    print(
        "\nPredictions saved to:",
        prediction_path,
    )


if __name__ == "__main__":
    main()

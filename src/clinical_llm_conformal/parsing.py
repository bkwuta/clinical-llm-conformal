import json
import re
from dataclasses import dataclass


@dataclass
class ClinicalPrediction:
    probability: float | None
    reasoning: str | None
    failed: bool
    raw_text: str


def parse_llm_response(text: str) -> ClinicalPrediction:
    if text is None:
        return ClinicalPrediction(None, None, True, "")

    raw = text.strip()

    if raw == "I do not know":
        return ClinicalPrediction(
            probability=None,
            reasoning="I do not know",
            failed=True,
            raw_text=raw,
        )

    candidates: list[str] = []

    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_match:
        candidates.append(fenced_match.group(1))

    object_match = re.search(
        r"\{.*\}",
        raw,
        flags=re.DOTALL,
    )
    if object_match:
        candidates.append(object_match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if "answer" not in obj:
            continue

        try:
            probability = float(obj["answer"])
        except (TypeError, ValueError):
            continue

        if not 0.0 <= probability <= 1.0:
            continue

        reasoning = obj.get("think")

        return ClinicalPrediction(
            probability=probability,
            reasoning=reasoning,
            failed=False,
            raw_text=raw,
        )

    return ClinicalPrediction(
        probability=None,
        reasoning=None,
        failed=True,
        raw_text=raw,
    )
"""Diagnosis Accuracy — predicted failure type vs the phase-3 reference labels.

    strict  accuracy = predicted == actual primary (priority convention)
    lenient accuracy = predicted ∈ actual failed checks (any deviation the labeler found)

Both are reported because the primary label is a dependency-order convention,
not a causal proof (phase-3 spot check: 75% strict / 90% lenient agreement with
human review). Diagnoses are made without gold; the labels are gold-derived and
live only on the evaluation side.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any


def score(diagnoses: dict[str, dict[str, Any]], labels: dict[str, dict[str, Any]]) -> tuple[dict, list[dict]]:
    rows, by_source = [], defaultdict(list)
    confusion: Counter = Counter()
    for cid, lab in labels.items():
        d = diagnoses.get(cid)
        if d is None:
            continue
        strict = d["failure_type"] == lab["actual_failure_type"]
        lenient = d["failure_type"] in lab["failed_checks"]
        rows.append({"case_id": cid, "predicted_failure_type": d["failure_type"],
                     "actual_failure_type": lab["actual_failure_type"],
                     "actual_failed_checks": lab["failed_checks"], "diagnosis_correct": strict,
                     "diagnosis_correct_lenient": lenient, "diagnosis_confidence": d["confidence"],
                     "diagnosis_source": d["source"]})
        by_source[d["source"]].append((strict, lenient))
        confusion[(lab["actual_failure_type"], d["failure_type"])] += 1
    n = len(rows)

    def rate(xs: list[bool]) -> str:
        return f"{sum(xs)}/{len(xs)} = {sum(xs) / len(xs):.1%}" if xs else "0/0"

    conf_ok = [r["diagnosis_confidence"] for r in rows if r["diagnosis_correct_lenient"]]
    conf_bad = [r["diagnosis_confidence"] for r in rows if not r["diagnosis_correct_lenient"]]
    summary = {
        "diagnosable_failures": n,
        "diagnosis_accuracy_strict": rate([r["diagnosis_correct"] for r in rows]),
        "diagnosis_accuracy_lenient": rate([r["diagnosis_correct_lenient"] for r in rows]),
        "by_source": {s: {"n": len(v), "strict": rate([a for a, _ in v]), "lenient": rate([b for _, b in v])}
                      for s, v in sorted(by_source.items())},
        "predicted_distribution": dict(Counter(r["predicted_failure_type"] for r in rows)),
        "actual_distribution": dict(Counter(r["actual_failure_type"] for r in rows)),
        "confusion_actual_to_predicted": {f"{a} -> {p}": c for (a, p), c in confusion.most_common()},
        "mean_confidence_when_right": round(statistics.mean(conf_ok), 3) if conf_ok else None,
        "mean_confidence_when_wrong": round(statistics.mean(conf_bad), 3) if conf_bad else None,
    }
    return summary, rows

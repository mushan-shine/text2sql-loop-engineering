"""Phase 0 qualification report: answers the five gating questions from evidence
files written by the phase-0 steps (runs/phase0/*.json). No numbers are typed in."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.beaver.compatibility import COMPATIBLE, REFERENCE_FAILED


def _load(d: Path, name: str) -> dict | None:
    p = d / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def decide_architecture(compat: dict) -> tuple[str, str]:
    """Map the measured compatibility onto the plan's cases A / B / C."""
    by = compat["by_status"]
    qualifiable = compat["cases"] - by.get(REFERENCE_FAILED, 0)
    ok = by.get(COMPATIBLE, 0)
    usable = compat.get("usable_with_adaptations", ok)
    if qualifiable and ok / qualifiable >= 0.95:
        return "A", f"{ok}/{qualifiable} qualifiable gold SQL reproduce the MySQL result unmodified"
    if qualifiable and usable / qualifiable >= 0.70:
        return "B", (f"{ok}/{qualifiable} compatible unmodified, {usable}/{qualifiable} usable with documented, "
                     "result-validated Benchmark Adapter rules; the rest are excluded and counted")
    return "C", (f"only {usable}/{qualifiable} usable — Execution Compatibility Boundary; keep BEAVER official "
                 "(MySQL) evaluation as reference and treat Databricks evaluation as a separate environment")


def build_report(runs_dir: str | Path = "runs/phase0") -> tuple[str, dict[str, Any]]:
    d = Path(runs_dir)
    env, imp, rep, compat, gold = (_load(d, n) for n in (
        "01_environment.json", "02_import.json", "02b_replicate.json", "03_compatibility.json",
        "04_gold_validation.json"))
    answers: dict[str, Any] = {}
    lines = ["# Phase 0 — BEAVER → Databricks Qualification Report", ""]

    if env:
        lines += ["## Environment", "", "```json", json.dumps(env, indent=2, ensure_ascii=False), "```", ""]

    lines += ["## Q1 BEAVER Dataset 是否成功进入 Databricks？", ""]
    if imp:
        answers["Q1"] = imp["cases"] > 0
        lines += [f"- cases imported: **{imp['cases']}** (split {imp['splits']}, dbs {imp['dbs']}); "
                  f"benchmark.cases {imp['cases_table']}",
                  f"- table metadata rows (beaver-table): {imp['tables_meta']}",
                  "- agent-facing view `benchmark.cases_agent_view` exposes only case_id / question / db", ""]
    else:
        lines += ["- NOT RUN", ""]

    lines += ["## Q2 BEAVER Schema 是否能够在 Databricks 中正确还原？", ""]
    if rep:
        rows_equal = rep["rows_mysql"] == rep["rows_databricks"]
        if rep["tables"] == 0 or rep["missing_dbs_in_mysql"] or not rows_equal:
            answers["Q2"] = "no"
        elif rep["tables"] == rep["tables_fidelity_ok"]:
            answers["Q2"] = "yes"
        else:  # every row present; some column profiles differ (see list)
            answers["Q2"] = "partial"
        lines += [f"- tables replicated: {rep['tables']}, fidelity OK: **{rep['tables_fidelity_ok']}**",
                  f"- rows MySQL / Databricks: {rep['rows_mysql']} / {rep['rows_databricks']}",
                  f"- dbs missing in MySQL: {rep['missing_dbs_in_mysql'] or 'none'}"]
        lines += [f"  - ✗ {f}" for f in rep["failed_tables"][:30]]
        lines += [""]
    else:
        lines += ["- NOT RUN", ""]

    lines += ["## Q3 BEAVER Gold SQL 有多少可以直接在 Databricks SQL 执行？", ""]
    if compat:
        arch, why = decide_architecture(compat)
        answers["Q3"] = {"executes_unmodified": compat["executes_unmodified_on_databricks"],
                         "compatible": compat["executes_and_matches_mysql"], "cases": compat["cases"],
                         "architecture": arch}
        lines += [f"- executes unmodified: **{compat['executes_unmodified_on_databricks']}/{compat['cases']}**",
                  f"- executes AND reproduces the MySQL (official engine) result: "
                  f"**{compat['executes_and_matches_mysql']}/{compat['cases']}**", "",
                  "| status | cases |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in compat["by_status"].items()]
        lines += ["", f"- adapter outcomes: {compat['adaptations'] or 'none attempted'}",
                  f"- result-equivalent adaptations by rule: {compat.get('adaptations_by_rule') or 'none'}",
                  f"- usable (compatible + result-equivalent adaptations): "
                  f"**{compat.get('usable_with_adaptations', compat['executes_and_matches_mysql'])}/{compat['cases']}**",
                  f"- static hazards (informational): {compat['static_hazards']}", "",
                  f"**Architecture decision: case {arch}** — {why}", ""]
    else:
        lines += ["- NOT RUN", ""]

    lines += ["## Q4 Gold SQL 的执行结果是否稳定？", ""]
    if compat and gold:
        recs = compat.get("records", [])
        ok_exec = [r for r in recs if r["execution_status"] == "SUCCESS"]
        unstable = [r["case_id"] for r in ok_exec if not r["databricks_stable"]]
        ref_unstable = [r["case_id"] for r in recs if r["reference_status"] == "SUCCESS" and not r["reference_stable"]]
        answers["Q4"] = not unstable and gold["drift"] == 0
        lines += [f"- repeated executions per engine within the run: all hashes equal except "
                  f"Databricks {len(unstable)} / MySQL {len(ref_unstable)} cases {unstable + ref_unstable}",
                  f"- re-execution in a separate session (step 04) vs qualified result: drift in **{gold['drift']}** cases",
                  f"- empty gold results: {gold['empty_gold_results']} (BEAVER counts both-empty as a match)", ""]
    else:
        lines += ["- NOT RUN", ""]

    lines += ["## Q5 是否能够在不修改 Benchmark Ground Truth 的情况下完成 Evaluation？", ""]
    if gold:
        e = gold["eligibility"]
        answers["Q5"] = e.get("PRIMARY", 0) > 0
        src = {}
        for r in gold.get("records", []):
            if r["evaluation_eligibility"] == "PRIMARY":
                k = (r["gold_source"] or "").split(":", 1)
                key = "original gold SQL" if k[0] == "databricks_original_gold_sql" else f"adapted ({k[1] if len(k) > 1 else '?'})"
                src[key] = src.get(key, 0) + 1
        lines += [f"- PRIMARY (result verified against MySQL, the official engine): **{e.get('PRIMARY', 0)}**"]
        lines += [f"  - {k}: {v}" for k, v in sorted(src.items())]
        lines += [f"- SECONDARY (validated adaptation, kept out of primary): {e.get('SECONDARY', 0)}",
                  f"- EXCLUDED (counted, not deleted): {e.get('EXCLUDED', 0)}",
                  "- gold SQL text in benchmark.cases is the original BEAVER text; adaptations live only in "
                  "benchmark.gold_adaptations", ""]
    else:
        lines += ["- NOT RUN", ""]

    lines += ["## Answers", "", "```json", json.dumps(answers, indent=2, ensure_ascii=False), "```"]
    return "\n".join(lines), answers

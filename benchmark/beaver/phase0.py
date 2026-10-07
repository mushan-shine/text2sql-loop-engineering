"""Phase 0 orchestration: environment → import → replicate → compatibility → gold.

Each step is idempotent and writes its evidence to Delta (and a local JSON copy
under ``runs/phase0/``) so the qualification report is reproducible.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.beaver import adapter, compatibility, replicate
from benchmark.beaver.dataset import BeaverCase
from benchmark.beaver.evaluator import CROSS_ENGINE_RULE, serialize_rows
from dbx import tables
from dbx.catalog import Layout, table_exists, upload_file, write_rows

log = logging.getLogger(__name__)

RUNS_DIR = Path("runs/phase0")


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def new_run_id(step: str) -> str:
    return f"{step}-{now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"


def save_local(name: str, payload: Any) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    p = RUNS_DIR / name
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- 02 import


def import_cases(dbx: Any, layout: Layout, cases: list[BeaverCase], tables_meta: dict[str, dict]) -> dict:
    """Write benchmark.cases once. Re-import must be byte-identical; never overwrite."""
    rows = [{**c.to_row(), "imported_at": now()} for c in cases]
    if table_exists(dbx, layout, layout.benchmark, "cases"):
        existing = dict(dbx.run(f"SELECT case_id, source_sha256 FROM {layout.fq(layout.benchmark, 'cases')}"))
        incoming = {r["case_id"]: r["source_sha256"] for r in rows}
        if existing != incoming:
            changed = sorted(k for k in incoming.keys() | existing.keys() if existing.get(k) != incoming.get(k))
            raise RuntimeError(
                f"benchmark.cases exists and differs for {len(changed)} cases (e.g. {changed[:3]}). "
                "Original benchmark data is never overwritten — drop the table deliberately if intended.")
        status = "unchanged (identical re-import)"
    else:
        write_rows(dbx, layout, layout.benchmark, "cases", rows, tables.CASES, mode="create_if_absent")
        status = "created"

    dbs = {c.db for c in cases}
    meta_rows = [
        {"db": t.get("db"), "table_name": name,
         "column_names": json.dumps(t.get("column_names"), ensure_ascii=False),
         "column_types": json.dumps(t.get("column_types"), ensure_ascii=False),
         "example_rows": json.dumps(t.get("example_rows"), ensure_ascii=False, default=str),
         "imported_at": now()}
        for name, t in tables_meta.items() if not dbs or t.get("db") in dbs
    ]
    write_rows(dbx, layout, layout.benchmark, "tables_meta", meta_rows, tables.TABLES_META, mode="overwrite")

    # Agent-facing view: setting=0 fields only. Gold columns are not selectable through it.
    dbx.run(f"CREATE OR REPLACE VIEW {layout.fq(layout.benchmark, 'cases_agent_view')} AS "
            f"SELECT case_id, question, db FROM {layout.fq(layout.benchmark, 'cases')}")
    summary = {"cases": len(rows), "cases_table": status, "tables_meta": len(meta_rows),
               "dbs": sorted(dbs), "splits": sorted({c.split for c in cases})}
    save_local("02_import.json", summary)
    return summary


# --------------------------------------------------------------------------- 02b replicate


def mysql_columns(mysql: Any, db: str, table: str) -> list[replicate.MySqlColumn]:
    rows = mysql.query(
        "SELECT COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, NUMERIC_PRECISION, NUMERIC_SCALE, COLLATION_NAME "
        "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION",
        (db, table))
    return [replicate.MySqlColumn(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]


def replicate_databases(mysql: Any, dbx: Any, layout: Layout, dbs: list[str], collation_policy: str,
                        only_tables: set[str] | None = None) -> dict:
    run_id = new_run_id("replicate")
    report_rows, missing_dbs = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for db in dbs:
            tbls = [r[0] for r in mysql.query(
                "SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s AND TABLE_TYPE='BASE TABLE'",
                (db,))]
            if not tbls:
                missing_dbs.append(db)
                continue
            dbx.run(f"CREATE SCHEMA IF NOT EXISTS `{layout.catalog}`.`{db}`")
            for t in sorted(tbls):
                if only_tables and t not in only_tables:
                    continue
                cols = mysql_columns(mysql, db, t)
                maps = [replicate.map_column(c, collation_policy) for c in cols]
                local = Path(tmp) / f"{db}.{t}.parquet"
                stats = replicate.write_parquet(mysql.iter_table(db, t), maps, local)
                vol = f"{layout.staging_root}/replica/{db}/{t}.parquet"
                upload_file(dbx.workspace_config, vol, str(local))
                fq = f"`{layout.catalog}`.`{db}`.{replicate.quote(t)}"
                dbx.run(replicate.create_table_sql(fq, maps))
                dbx.run(replicate.insert_from_parquet_sql(fq, maps, vol))
                my_prof = mysql.query(replicate.profile_sql(f"`{db}`.`{t}`", maps, "mysql"))[0]
                dx_prof = dbx.run(replicate.profile_sql(fq, maps, "databricks"))[0]
                fid = replicate.TableFidelity(
                    db, t, int(my_prof[0]), int(dx_prof[0]), len(maps),
                    replicate.compare_profiles(maps, my_prof, dx_prof, stats.nulled_invalid_dates),
                    stats.nulled_invalid_dates, sorted({m.note for m in maps if m.note}))
                log.info("%s.%s rows %d/%d ok=%s", db, t, fid.mysql_rows, fid.databricks_rows, fid.ok)
                report_rows.append({
                    "run_id": run_id, "db": db, "table_name": t, "mysql_rows": fid.mysql_rows,
                    "databricks_rows": fid.databricks_rows, "columns": fid.columns,
                    "column_types": json.dumps({m.name: [m.mysql_type, m.databricks_type] for m in maps}),
                    "mismatched_columns": json.dumps(fid.mismatched_columns),
                    "nulled_invalid_dates": fid.nulled_invalid_dates, "notes": "; ".join(fid.notes),
                    "fidelity_ok": fid.ok, "created_at": now()})
    if report_rows:
        write_rows(dbx, layout, layout.benchmark, "replication_report", report_rows, tables.REPLICATION)
    summary = {
        "run_id": run_id, "dbs": dbs, "missing_dbs_in_mysql": missing_dbs,
        "tables": len(report_rows), "tables_fidelity_ok": sum(r["fidelity_ok"] for r in report_rows),
        "rows_mysql": sum(r["mysql_rows"] for r in report_rows),
        "rows_databricks": sum(r["databricks_rows"] for r in report_rows),
        "failed_tables": [f"{r['db']}.{r['table_name']}: {r['mismatched_columns']}"
                          for r in report_rows if not r["fidelity_ok"]],
    }
    save_local("02b_replicate.json", {**summary, "tables_detail": report_rows})
    return summary


# --------------------------------------------------------------------------- 03 compatibility


@dataclass
class QualificationConfig:
    repeats: int = 3
    try_adapter: bool = True
    admit_validated_adaptations: bool = False
    environment: str = "{}"


def run_compatibility(cases: list[BeaverCase], mysql: Any, dbx: Any, layout: Layout,
                      qc: QualificationConfig) -> dict:
    run_id = new_run_id("compat")
    compat_rows, adapt_rows = [], []
    for i, case in enumerate(cases, 1):
        rec, ref_runs, _ = compatibility.validate_case(case, mysql, dbx, qc.repeats)
        log.info("[%d/%d] %s → %s (%s)", i, len(cases), case.case_id, rec.compatibility_status, rec.reason)
        env = json.dumps({**json.loads(qc.environment), "comparison": CROSS_ENGINE_RULE})
        compat_rows.append({**rec.to_row(), "run_id": run_id, "environment": env, "created_at": now()})
        if qc.try_adapter and rec.compatibility_status not in (compatibility.COMPATIBLE,
                                                               compatibility.REFERENCE_FAILED):
            ad = adapter.try_adapt(case, ref_runs, dbx, qc.repeats)
            adapt_rows.append({**ad.to_row(), "run_id": run_id, "created_at": now()})
    write_rows(dbx, layout, layout.benchmark, "sql_compatibility", compat_rows, tables.COMPATIBILITY)
    if adapt_rows:
        write_rows(dbx, layout, layout.benchmark, "gold_adaptations", adapt_rows, tables.ADAPTATIONS)
    summary = summarize_compatibility(compat_rows, adapt_rows)
    summary["run_id"] = run_id
    save_local("03_compatibility.json", {**summary, "records": compat_rows, "adaptation_records": adapt_rows})
    return summary


def summarize_compatibility(compat_rows: list[dict], adapt_rows: list[dict]) -> dict:
    by_status = Counter(r["compatibility_status"] for r in compat_rows)
    n = len(compat_rows)
    executable = sum(r["execution_status"] == "SUCCESS" for r in compat_rows)
    equivalent = [a for a in adapt_rows if a["semantic_validation"] == adapter.RESULT_EQUIVALENT]
    return {
        "cases": n,
        "by_status": {s: by_status.get(s, 0) for s in compatibility.STATUSES},
        "executes_unmodified_on_databricks": executable,
        "executes_and_matches_mysql": by_status.get(compatibility.COMPATIBLE, 0),
        "compatible_rate": round(by_status.get(compatibility.COMPATIBLE, 0) / n, 4) if n else None,
        "adaptations": dict(Counter(a["semantic_validation"] for a in adapt_rows)),
        "adaptations_by_rule": dict(Counter(a["adaptation_rule"] for a in equivalent)),
        "usable_with_adaptations": by_status.get(compatibility.COMPATIBLE, 0) + len(equivalent),
        "static_hazards": dict(Counter(h for r in compat_rows for h in filter(None, r["static_hazards"].split(","))))
    }


# --------------------------------------------------------------------------- 04 gold


def build_gold_results(cases: list[BeaverCase], dbx: Any, layout: Layout, compat_run_id: str,
                       qc: QualificationConfig) -> dict:
    """Re-execute the qualified SQL on Databricks in a fresh session and freeze it as
    benchmark.gold_results.

    Equivalence with MySQL (the official engine) was established in the
    compatibility run under the cross-engine comparison rule; here the result
    must reproduce the Databricks result hash recorded in that run (drift check).
    """
    fq_c = layout.fq(layout.benchmark, "sql_compatibility")
    fq_a = layout.fq(layout.benchmark, "gold_adaptations")
    compat = {r[0]: r[1:] for r in dbx.run(
        f"SELECT case_id, compatibility_status, reference_result_hash, reason, result_hash "
        f"FROM {fq_c} WHERE run_id = '{compat_run_id}'")}
    if not compat:
        raise RuntimeError(f"no compatibility records for run {compat_run_id}")
    adapted: dict[str, tuple[str, str, str]] = {}
    if table_exists(dbx, layout, layout.benchmark, "gold_adaptations"):
        adapted = {r[0]: (r[1], r[2], r[3]) for r in dbx.run(
            f"SELECT case_id, adapted_sql, adapted_result_hash, adaptation_rule FROM {fq_a} "
            f"WHERE run_id = '{compat_run_id}' AND semantic_validation = '{adapter.RESULT_EQUIVALENT}'")}
    run_id = new_run_id("gold")
    out = []
    for case in cases:
        status, ref_hash, reason, dbx_hash = compat.get(case.case_id, (None, None, "not in compatibility run", None))
        base = {"case_id": case.case_id, "db": case.db, "gold_sql": case.gold_sql,
                "verified_against": f"mysql (BEAVER official engine); {CROSS_ENGINE_RULE}",
                "reference_result_hash": ref_hash,
                "qualification_run_id": compat_run_id, "created_at": now()}
        if status == compatibility.COMPATIBLE:
            sql, source, eligibility, expected = case.gold_sql, "databricks_original_gold_sql", "PRIMARY", dbx_hash
        elif case.case_id in adapted:
            sql, expected, rule = adapted[case.case_id]
            source = f"databricks_adapted_sql:{rule}"
            eligibility = "PRIMARY" if qc.admit_validated_adaptations else "SECONDARY"
        else:
            out.append({**base, "executed_sql": None, "gold_result": None, "result_hash": None, "row_count": None,
                        "column_names": None, "execution_status": None, "execution_time": None,
                        "gold_source": None, "evaluation_eligibility": "EXCLUDED",
                        "exclusion_reason": f"excluded_due_to_execution_incompatibility: {status}: {reason}"})
            continue
        runs = compatibility.run_repeated(dbx, sql, case.db, qc.repeats)
        h = runs.result_hashes[0] if runs.result_hashes else None
        drift = runs.status != "SUCCESS" or not runs.stable or h != expected
        out.append({**base, "executed_sql": sql,
                    "gold_result": serialize_rows(runs.rows) if not drift else None,
                    "result_hash": h, "row_count": runs.row_count,
                    "column_names": json.dumps(runs.columns) if runs.columns else None,
                    "execution_status": runs.status,
                    "execution_time": int(sum(runs.elapsed_ms) / len(runs.elapsed_ms)) if runs.elapsed_ms else None,
                    "gold_source": source,
                    "evaluation_eligibility": "EXCLUDED" if drift else eligibility,
                    "exclusion_reason": ("gold_result_drift: re-execution does not reproduce the result "
                                         "qualified in the compatibility run") if drift else None})
    write_rows(dbx, layout, layout.benchmark, "gold_results", out, tables.GOLD_RESULTS, mode="overwrite")
    elig = Counter(r["evaluation_eligibility"] for r in out)
    summary = {"run_id": run_id, "compat_run_id": compat_run_id, "cases": len(out),
               "eligibility": dict(elig),
               "drift": sum(1 for r in out if (r["exclusion_reason"] or "").startswith("gold_result_drift")),
               "empty_gold_results": sum(1 for r in out if r["row_count"] == 0)}
    save_local("04_gold_validation.json", {**summary, "records": [
        {k: v for k, v in r.items() if k != "gold_result"} for r in out]})
    return summary

"""Data access for the Loop Debug Console — reads the Delta tables written by the
pipeline (benchmark.*, traces.*, evaluation.*). Nothing is computed from local
files, so new phase-7 / phase-8 runs appear as soon as they are published.

Auth: locally a ~/.databrickscfg profile (SHT_DATABRICKS_PROFILE, default
DEFAULT); inside Databricks Apps the app's own credentials plus
DATABRICKS_WAREHOUSE_ID.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATALOG = os.environ.get("SHT_CATALOG", "text2sql_loop")


def connect():
    from execution.databricks_sql import DatabricksSqlExecutor

    in_app = bool(os.environ.get("DATABRICKS_APP_NAME") or os.environ.get("DATABRICKS_CLIENT_ID"))
    profile = None if in_app else os.environ.get("SHT_DATABRICKS_PROFILE", "DEFAULT")
    wh = os.environ.get("DATABRICKS_WAREHOUSE_ID")
    return DatabricksSqlExecutor(catalog=CATALOG, profile=profile,
                                 http_path=f"/sql/1.0/warehouses/{wh}" if wh else None)


def t(schema: str, table: str) -> str:
    return f"`{CATALOG}`.`{schema}`.`{table}`"


# ------------------------------------------------------------------ queries

def runs(conn) -> pd.DataFrame:
    df = conn.query_df(f"SELECT * FROM {t('evaluation', 'runs')} ORDER BY created_at")
    df["summary"] = df["summary_json"].map(json.loads)
    df["meta"] = df["meta_json"].map(json.loads)
    df["arm"] = df["meta"].map(lambda m: m.get("arm"))
    df["strategy"] = df["meta"].map(lambda m: m.get("strategy"))
    df["verifier"] = df["meta"].map(lambda m: m.get("verifier"))
    df["policy"] = df["meta"].map(lambda m: m.get("policy") or "targeted")
    df["disabled"] = df["meta"].map(lambda m: m.get("disabled") or "")
    df["source"] = df["meta"].map(lambda m: m.get("source") or "cli")  # console = started from the Run page
    return df


def traces(conn, run_id: str) -> pd.DataFrame:
    return conn.query_df(f"SELECT * FROM {t('traces', 'execution_traces')} WHERE run_id = '{run_id}' "
                         "ORDER BY case_id, attempt_id")


def evaluations(conn, run_id: str) -> pd.DataFrame:
    return conn.query_df(f"SELECT case_id, attempt_id, correct, eval_message FROM "
                         f"{t('evaluation', 'evaluation_results')} WHERE run_id = '{run_id}'")


def failure_labels(conn) -> pd.DataFrame:
    return conn.query_df(f"SELECT * FROM {t('evaluation', 'failure_labels')}")


def diagnosis_eval(conn) -> pd.DataFrame:
    return conn.query_df(f"SELECT * FROM {t('evaluation', 'diagnosis_eval')}")


def benchmark_summary(conn) -> dict:
    g = conn.query_df(f"SELECT evaluation_eligibility AS e, gold_source AS s, count(*) AS n FROM "
                      f"{t('benchmark', 'gold_results')} GROUP BY ALL")
    c = conn.query_df(f"SELECT count(*) AS n FROM {t('benchmark', 'cases')}")
    r = conn.query_df(f"SELECT count(*) AS tables, sum(mysql_rows) AS rows, sum(CAST(fidelity_ok AS INT)) AS ok "
                      f"FROM {t('benchmark', 'replication_report')} WHERE run_id = (SELECT max(run_id) FROM "
                      f"{t('benchmark', 'replication_report')})")
    return {"gold": g, "cases": int(c["n"][0]), "replication": r.iloc[0].to_dict()}

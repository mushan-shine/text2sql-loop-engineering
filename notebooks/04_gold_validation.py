# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Gold result validation
# MAGIC `benchmark.gold_results` freezes the ground truth. Each PRIMARY row is the
# MAGIC Databricks result of the **original** gold SQL whose hash equals the MySQL
# MAGIC (official engine) result. Later experiments compare against this frozen
# MAGIC result instead of re-running gold SQL.
# MAGIC
# MAGIC This notebook re-executes the qualified SQL **inside the Databricks runtime**
# MAGIC (serverless notebook compute — a different path from the SQL Warehouse used by
# MAGIC the workstation run) and checks that the frozen hashes still reproduce.

# COMMAND ----------

# MAGIC %pip install sqlglot pyyaml
# MAGIC %restart_python

# COMMAND ----------

import os
import sys

dbutils.widgets.text("catalog", "text2sql_loop")
dbutils.widgets.text("repo_root", "")  # workspace path of the repo (Git folder); default: parent of notebooks/
catalog = dbutils.widgets.get("catalog")
root = dbutils.widgets.get("repo_root") or os.path.dirname(os.getcwd())
sys.path.insert(0, root)

from benchmark.beaver.compatibility import run_repeated  # noqa: E402
from execution.databricks_sql import SparkSqlExecutor  # noqa: E402

B = f"`{catalog}`.`benchmark`"

# COMMAND ----------

display(spark.sql(f"""
  SELECT evaluation_eligibility, gold_source, count(*) AS cases, sum(CAST(row_count = 0 AS INT)) AS empty_results
  FROM {B}.gold_results GROUP BY ALL ORDER BY evaluation_eligibility"""))

# COMMAND ----------

ex = SparkSqlExecutor(spark, catalog, ansi_mode=False)
rows = spark.sql(f"""SELECT case_id, db, executed_sql, result_hash FROM {B}.gold_results
                     WHERE evaluation_eligibility <> 'EXCLUDED'""").collect()
checks = []
for r in rows:
    runs = run_repeated(ex, r.executed_sql, r.db, repeats=2)
    h = runs.result_hashes[0] if runs.result_hashes else None
    checks.append((r.case_id, runs.status, runs.stable, h == r.result_hash, runs.error_class))
df = spark.createDataFrame(
    checks, "case_id string, status string, stable boolean, reproduces_frozen boolean, error_class string")
display(df.groupBy("status", "stable", "reproduces_frozen").count())

# COMMAND ----------

display(df.where("NOT reproduces_frozen OR NOT stable"))

# COMMAND ----------

display(spark.sql(f"SELECT case_id, exclusion_reason FROM {B}.gold_results WHERE evaluation_eligibility = 'EXCLUDED'"))

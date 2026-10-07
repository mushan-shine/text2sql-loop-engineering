# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · BEAVER import
# MAGIC Written by `python scripts/phase0.py import` and `... replicate`:
# MAGIC * `benchmark.cases` — sampled BEAVER cases, **original gold SQL**, `source_sha256` fingerprint
# MAGIC * `benchmark.cases_agent_view` — the only view an agent may read (case_id, question, db)
# MAGIC * `benchmark.tables_meta` — beaver-table schema metadata (setting=0 context)
# MAGIC * `<catalog>.<db>.*` — BEAVER databases restored from the MySQL dump
# MAGIC * `benchmark.replication_report` — per-table fidelity (row counts, column profiles)

# COMMAND ----------

dbutils.widgets.text("catalog", "text2sql_loop")
catalog = dbutils.widgets.get("catalog")
B = f"`{catalog}`.`benchmark`"

# COMMAND ----------

display(spark.sql(f"""
  SELECT split, db, count(*) AS cases, count(DISTINCT source_sha256) AS distinct_fingerprints,
         sum(CAST(contains_domain_knowledge AS INT)) AS with_domain_knowledge
  FROM {B}.cases GROUP BY split, db"""))

# COMMAND ----------

# Gold columns must not be reachable through the agent view.
agent_cols = spark.table(f"{catalog}.benchmark.cases_agent_view").columns
assert agent_cols == ["case_id", "question", "db"], agent_cols
agent_cols

# COMMAND ----------

display(spark.sql(f"""
  SELECT run_id, count(*) AS tables, sum(CAST(fidelity_ok AS INT)) AS fidelity_ok,
         sum(mysql_rows) AS mysql_rows, sum(databricks_rows) AS databricks_rows,
         sum(nulled_invalid_dates) AS nulled_invalid_dates
  FROM {B}.replication_report GROUP BY run_id ORDER BY run_id DESC"""))

# COMMAND ----------

display(spark.sql(f"SELECT db, table_name, mismatched_columns, notes FROM {B}.replication_report WHERE NOT fidelity_ok"))

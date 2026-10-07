# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · SQL compatibility
# MAGIC `python scripts/phase0.py compat` ran every **original** BEAVER gold SQL on MySQL
# MAGIC (BEAVER's official engine) and, unmodified, on Databricks SQL — each several
# MAGIC times with the result cache off. `COMPATIBLE` = executes **and** reproduces the
# MAGIC MySQL result **and** is stable. Executing without error alone is not enough.

# COMMAND ----------

dbutils.widgets.text("catalog", "text2sql_loop")
dbutils.widgets.text("run_id", "")
catalog = dbutils.widgets.get("catalog")
B = f"`{catalog}`.`benchmark`"
run_id = dbutils.widgets.get("run_id") or spark.sql(
    f"SELECT max_by(run_id, created_at) FROM {B}.sql_compatibility").first()[0]
run_id

# COMMAND ----------

display(spark.sql(f"""
  SELECT compatibility_status, count(*) AS cases,
         sum(CAST(execution_status = 'SUCCESS' AS INT)) AS executes_on_databricks
  FROM {B}.sql_compatibility WHERE run_id = '{run_id}'
  GROUP BY compatibility_status ORDER BY cases DESC"""))

# COMMAND ----------

display(spark.sql(f"""
  SELECT case_id, compatibility_status, reason, execution_error_class, static_hazards,
         reference_row_count, row_count, set_match, multiset_match
  FROM {B}.sql_compatibility
  WHERE run_id = '{run_id}' AND compatibility_status <> 'COMPATIBLE' ORDER BY compatibility_status"""))

# COMMAND ----------

# MAGIC %md Static hazards vs outcome — which MySQL constructs actually cause drift.

# COMMAND ----------

display(spark.sql(f"""
  SELECT hazard, compatibility_status, count(*) AS cases
  FROM {B}.sql_compatibility LATERAL VIEW explode(split(static_hazards, ',')) h AS hazard
  WHERE run_id = '{run_id}' AND hazard <> ''
  GROUP BY hazard, compatibility_status ORDER BY hazard"""))

# COMMAND ----------

# MAGIC %md Benchmark Adapter records (the original gold SQL stays untouched in `benchmark.cases`).

# COMMAND ----------

if spark.catalog.tableExists(f"{catalog}.benchmark.gold_adaptations"):
    display(spark.sql(f"""SELECT case_id, adaptation_rule, semantic_validation, detail, original_gold_sql, adapted_sql
                          FROM {B}.gold_adaptations WHERE run_id = '{run_id}'"""))

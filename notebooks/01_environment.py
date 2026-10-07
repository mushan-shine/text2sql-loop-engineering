# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Environment
# MAGIC Unity Catalog layout for the Self-Healing Text-to-SQL project and the execution
# MAGIC environment every experiment runs under. The same settings are applied by
# MAGIC `execution/databricks_sql.py` when phase 0 runs from a workstation.
# MAGIC
# MAGIC Phase 0 needs BEAVER's official engine (MySQL) as a reference oracle, so the
# MAGIC data-moving steps run from a workstation: `python scripts/phase0.py all`.
# MAGIC These notebooks inspect and re-check the evidence inside Databricks.

# COMMAND ----------

dbutils.widgets.text("catalog", "text2sql_loop")
dbutils.widgets.dropdown("ansi_mode", "false", ["false", "true"])
catalog = dbutils.widgets.get("catalog")
ansi_mode = dbutils.widgets.get("ansi_mode") == "true"

# COMMAND ----------

for schema in ("benchmark", "traces", "evaluation"):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`benchmark`.`staging`")
display(spark.sql(f"SHOW SCHEMAS IN `{catalog}`"))

# COMMAND ----------

# MAGIC %md
# MAGIC Environment checks: runtime version, ANSI mode (MySQL returns NULL on `x/0`,
# MAGIC ANSI Databricks raises) and UTF8_LCASE collation (mirrors MySQL `_ci` columns).

# COMMAND ----------

spark.sql(f"SET ANSI_MODE = {str(ansi_mode).lower()}")
checks = {
    "version": spark.sql("SELECT current_version()").first()[0],
    "ansi_mode": spark.conf.get("spark.sql.ansi.enabled"),
    "div_by_zero": spark.sql("SELECT 1/0").first()[0] if not ansi_mode else "raises",
    "lcase_collation": spark.sql("SELECT 'A' COLLATE UTF8_LCASE = 'a'").first()[0],
}
checks

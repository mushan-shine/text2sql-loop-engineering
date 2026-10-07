# Databricks notebook source
# MAGIC %md
# MAGIC # 步骤 8：UNION 排序规则修复、修复结果重复时提前结束
# MAGIC
# MAGIC **现象**：开发集有一道题 5 次尝试都报同一个错：
# MAGIC `[INCOMPATIBLE_COLUMN_TYPE] UNION ... "STRING" type which is not compatible with "STRING COLLATE UTF8_LCASE"`。
# MAGIC
# MAGIC **原因**：步骤 1 把字符串列建成了 `STRING COLLATE UTF8_LCASE`。模型在 UNION 里用 `CAST(NULL AS STRING)` 补空列，
# MAGIC 普通 STRING 和 UTF8_LCASE 列不兼容。诊断只把它归为通用执行错误，模型 4 轮都没改对，第 3–5 次的 SQL 完全相同。
# MAGIC
# MAGIC **改动**：
# MAGIC - 生成规则第 7 条（prompt 版本 `baseline-v2.1` / `baseline-v3.1-dynfs`）；
# MAGIC - 诊断规则 `collation_mismatch`，RepairSQL 先确定性地把 `CAST(NULL AS STRING)` 换成 `NULL`，不调 LLM；
# MAGIC - 修复后的 SQL 和之前某次尝试相同时提前结束。

# COMMAND ----------

# MAGIC %pip install -q -r requirements-notebook.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 先在数据库里看清楚原因
# MAGIC
# MAGIC 同一个 UNION，四种写法：`CAST(NULL AS STRING)` 和字符串字面量会报错，裸 `NULL` 和带 `COLLATE UTF8_LCASE` 的字面量可以。

# COMMAND ----------

from dbx.runtime import project_catalog

catalog = project_catalog()
for label, tail in [("CAST(NULL AS STRING)", "CAST(NULL AS STRING)"), ("字符串字面量", "'x'"),
                    ("裸 NULL", "NULL"), ("字面量 + COLLATE", "'x' COLLATE UTF8_LCASE")]:
    try:
        spark.sql(f"SELECT DEPARTMENT_NAME FROM `{catalog}`.dw.sis_department UNION ALL SELECT {tail}").limit(1).collect()
        print(f"{label:24s} 可以执行")
    except Exception as e:
        print(f"{label:24s} 报错：{str(e).splitlines()[0][:90]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 测试：不调 LLM 修好排序规则错误；修复结果重复时提前结束

# COMMAND ----------

assert run_tests("tests/test_phase6.py", "-k", "collation or repeats") == 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 真实运行（可选，会产生费用）
# MAGIC
# MAGIC 用 DeepSeek 在开发集上跑一遍（约 65 万 token），能看到那道题被确定性修复。需要在步骤 0 保存过 DeepSeek 的 key。
# MAGIC
# MAGIC 为了避免点 Run all 时意外产生费用，这一格默认跳过：在 notebook 顶部的 `run_deepseek` 输入框选“是”，再运行这一格。

# COMMAND ----------

dbutils.widgets.dropdown("run_deepseek", "否", ["否", "是"])
if dbutils.widgets.get("run_deepseek") == "是":
    os.environ["LLM_PROVIDER"] = "deepseek"
    try:
        run_script("phase6.py", "--few-shot", "dynamic", "--knowledge", "on", "--max-repairs", "4")
    finally:
        os.environ.pop("LLM_PROVIDER", None)
    show_summary(latest_run("phase6"))
else:
    print("已跳过 DeepSeek 运行（会产生费用）。需要时在顶部 run_deepseek 选“是”后再运行这一格。")

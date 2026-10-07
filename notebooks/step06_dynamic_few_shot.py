# Databricks notebook source
# MAGIC %md
# MAGIC # 步骤 6：相似题示例
# MAGIC
# MAGIC 自检发现不了“能执行但答错”，那就从源头减少这类错误。错误分析显示：能执行但答错的题，大多是用的表和标准答案不同，
# MAGIC 而且漏用的表多数已经在检索出的候选表里——问题不在检索，而在模型从一组相似的表里选错。
# MAGIC
# MAGIC 做法（`agent/examples.py`）：从训练集的已解题（排除评测集和开发集）中，按问题文本检索最相似的 4 道题替换固定示例，
# MAGIC 并把这些示例用到的表（最多 6 张）补进 schema。

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-notebook.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 单次生成：固定示例 vs 相似题示例
# MAGIC
# MAGIC 参考（glm）：答对 0 → 3，能执行 4 → 16。

# COMMAND ----------

run_script("phase1.py", "dev", "--few-shot", "dynamic")

# COMMAND ----------

show_summary(latest_run("phase1", "baseline-", mode="dev"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 加上内循环，并做错误分析
# MAGIC
# MAGIC `analyze_run.py` 从 Delta 读取一次已发布的运行，逐题对照标准答案：用表、列、关联键、过滤值、统计运算、行数和列数。

# COMMAND ----------

run_script("phase6.py", "--few-shot", "dynamic", "--max-repairs", "1")
loop_run = latest_run("phase6")
run_script("publish_run.py", loop_run, "--experiment-id", "targeted_loop")

# COMMAND ----------

run_script("analyze_run.py", loop_run.rsplit("/", 1)[-1])

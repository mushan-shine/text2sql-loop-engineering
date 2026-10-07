# Databricks notebook source
# MAGIC %md
# MAGIC # 步骤 3：可观测与评测分析
# MAGIC
# MAGIC 基线跑出来后，要能看清每道题发生了什么（trace），以及错在哪里、为什么错：
# MAGIC - **发布**（`dbx/publish.py`）：每次尝试一行写进 `traces.execution_traces`；用标准答案算出的对错写进 `evaluation.*`，两者分开存放，后面的内循环只能读 trace，看不到答案；汇总写进 `evaluation.runs` 和 MLflow；
# MAGIC - **失败标注器**（`benchmark/beaver/subtasks.py`，只用于离线评测）：把生成的 SQL 和标准 SQL 解析成语法树逐项比较，标出主因；
# MAGIC - **干预实验**（`evaluation/intervention.py`）：每次只把标准答案的一类信息提示给模型，区分“缺信息”和“模型能力不够”。

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-notebook.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

dev_run = latest_run("phase1", "baseline-", mode="dev")
run_id = dev_run.rsplit("/", 1)[-1]
print(dev_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 发布到 Delta 和 MLflow

# COMMAND ----------

run_script("publish_run.py", dev_run)

# COMMAND ----------

from dbx.runtime import project_catalog

catalog = project_catalog()
display(spark.sql(f"""SELECT case_id, attempt_id, execution_status, left(generated_sql, 120) AS sql
                      FROM `{catalog}`.traces.execution_traces WHERE run_id = '{run_id}' ORDER BY case_id"""))
display(spark.sql(f"""SELECT case_id, attempt_id, correct, eval_table_recall
                      FROM `{catalog}`.evaluation.evaluation_results WHERE run_id = '{run_id}' ORDER BY case_id"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 失败标注：错在哪里

# COMMAND ----------

run_script("phase3.py", "label", dev_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 干预实验（可选，约 160 次模型调用，glm 约 40 分钟）
# MAGIC
# MAGIC 参考：glm 即使拿到标准答案的全部信息也修不好（0/30），瓶颈是模型能力；DeepSeek 能修好 7/27，说明对它来说信息不足才是主要问题。

# COMMAND ----------

run_script("phase3.py", "intervene", dev_run)

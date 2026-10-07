# Databricks notebook source
# MAGIC %md
# MAGIC # 步骤 2：基础版（检索 + 单次生成）
# MAGIC
# MAGIC 做出最简单、能端到端运行的 Text-to-SQL，作为后面所有优化的对照基线：
# MAGIC - **检索**（`agent/retriever.py`）：BM25 从 97 张表中选出 20 张候选表；
# MAGIC - **生成**（`agent/generator.py`）：prompt = 规则 + 候选表 schema + 固定 3 个示例 + 问题，只生成一次；
# MAGIC - **判分**（`evaluation/baseline.py`）：生成结束后才用标准答案判分。
# MAGIC
# MAGIC 数据集：开发集 30 题（所有调参只在这里做）、评测集 89 题（配置冻结后才运行）。

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-notebook.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 开发集基线
# MAGIC
# MAGIC 默认模型 glm-4-flash（免费），每题约 15 秒。结束时打印汇总：`first_correct`（答对）、`executable_first`（能执行）、`table_recall`（表召回率）。
# MAGIC 参考：答对 0、能执行 4、表召回约 0.92。

# COMMAND ----------

run_script("phase1.py", "dev")

# COMMAND ----------

dev_run = latest_run("phase1", "baseline-", mode="dev")
print(dev_run)
for line in (REPO / dev_run / "results.jsonl").read_text(encoding="utf-8").splitlines()[:3]:
    r = json.loads(line)
    print(r["case_id"], r["execution_status"], "答对" if r["correct"] else "答错")
    print(r["generated_sql"][:600], "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 评测集基线（配置冻结后的正式结果，可选）
# MAGIC
# MAGIC 89 题，约 25 分钟。参考：答对 2、能执行 21。评测集只在配置冻结后运行，不要根据它的结果调参。

# COMMAND ----------

run_script("phase1.py", "baseline")

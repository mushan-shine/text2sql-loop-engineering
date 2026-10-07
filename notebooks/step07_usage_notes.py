# Databricks notebook source
# MAGIC %md
# MAGIC # 步骤 7：数仓使用说明
# MAGIC
# MAGIC 相似题示例只覆盖和新问题字面相近的几道题。这一步把整个训练集里的用表习惯统计出来，作为通用知识补充给模型
# MAGIC （`agent/knowledge.py`、`scripts/build_knowledge.py`，只用训练集，不含任何评测集和开发集的信息）：
# MAGIC - 概念 → 常用表：问题里出现某个词时，已解题通常用哪些表（IDF 加权）；
# MAGIC - 相似表组：列名相似的表，以及在同类问题里各自的使用比例；
# MAGIC - 关联约定：每对表常用的关联键，以及 INNER / LEFT JOIN 的比例。
# MAGIC
# MAGIC 运行时每道题只取相关的几行放进 prompt。

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-notebook.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 从训练集统计使用说明

# COMMAND ----------

run_script("build_knowledge.py")

# COMMAND ----------

from agent.knowledge import load_knowledge

kb = load_knowledge(REPO / "runs/knowledge/kb.json")
question = "List the subjects offered in fall 2023 and the number of students enrolled in each"
from agent.retriever import BM25TableRetriever, SchemaCatalog
catalog = SchemaCatalog.from_json((REPO / "runs/phase1/schema_dw.json").read_text(encoding="utf-8"))
tables = BM25TableRetriever(catalog).retrieve(question, 20).tables
print(kb.notes_for(question, tables))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 单次生成：相似题示例 vs 相似题示例 + 使用说明
# MAGIC
# MAGIC 参考（glm）：答对 3 → 5，能执行仍为 16，token 只增加约 3%。

# COMMAND ----------

run_script("phase1.py", "dev", "--few-shot", "dynamic", "--knowledge", "on")
show_summary(latest_run("phase1", "baseline-", mode="dev"))

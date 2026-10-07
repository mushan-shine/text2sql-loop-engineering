# Databricks notebook source
# MAGIC %md
# MAGIC # 步骤 1：数据底座
# MAGIC
# MAGIC **要解决的问题**：BEAVER 官方用 MySQL 执行标准 SQL，本项目跑在 Databricks 上。换了引擎，标准答案还成不成立？
# MAGIC
# MAGIC 完整的做法（`benchmark/beaver/`、`scripts/phase0.py`）需要本地 MySQL：
# MAGIC 1. 把 BEAVER 的 97 张表从 MySQL 复制到 Delta，字符串列使用 `UTF8_LCASE` 排序规则（和 MySQL 一样不区分大小写）；
# MAGIC 2. 每条标准 SQL 在两个引擎上各执行 3 次并比较结果；
# MAGIC 3. 不一致的查明原因，只加有文档依据、并逐题验证过的改写规则；
# MAGIC 4. 冻结标准答案到 `benchmark.gold_results`。
# MAGIC
# MAGIC 结果：严格比较只有 37/100 一致；查明三类方言差异、补两条改写规则后，**89 题可用**（原样 46 + 改写 43）。
# MAGIC
# MAGIC 这一步的产出已经打包成**课程数据包**（BEAVER 为 MIT 许可，数据包注明了出处）。
# MAGIC 本 notebook 把数据包导入你自己的工作区，然后查看第 1 步的比对结果。
# MAGIC 想完整体验双引擎比对的同学，可以在本地安装 MySQL 后按 README 运行 `scripts/phase0.py`。

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-notebook.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 导入课程数据包
# MAGIC
# MAGIC 从 HuggingFace 下载数据包（约 55 MB），在你的工作区里建 catalog、schema 和表，然后导入数据。
# MAGIC 建表时保留每一列原来的类型和排序规则。第一次运行约需 10–15 分钟；已经导入过的表会自动跳过。

# COMMAND ----------

# 讲师离线测试：在 notebook 顶部建一个名为 course_data_dir 的输入框填数据包目录，即改为从该目录导入
try:
    os.environ["SHT_COURSE_DATA_DIR"] = dbutils.widgets.get("course_data_dir")
except Exception:
    pass
run_script("load_course_data.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 检查导入结果
# MAGIC
# MAGIC 预期：`dw` 有 97 张表、共 421,707 行；字符串列的类型是 `string collate UTF8_LCASE`。

# COMMAND ----------

from dbx.runtime import project_catalog

catalog = project_catalog()
display(spark.sql(f"SHOW TABLES IN `{catalog}`.`dw`"))
display(spark.sql(f"""
    SELECT 'dw 表数' AS item, COUNT(*) AS value FROM `{catalog}`.information_schema.tables WHERE table_schema = 'dw'
    UNION ALL
    SELECT '标准答案（PRIMARY）', COUNT(*) FROM `{catalog}`.benchmark.gold_results WHERE evaluation_eligibility = 'PRIMARY'
"""))
display(spark.sql(f"DESCRIBE TABLE `{catalog}`.dw.sis_department"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 第 1 步的比对报告
# MAGIC
# MAGIC 由数据包里的比对结果（`runs/phase0/`）生成，回答 5 个问题：数据是否进入 Databricks、表结构是否正确还原、
# MAGIC 多少标准 SQL 能直接执行、结果是否稳定、能否在不修改标准答案的前提下完成评测。

# COMMAND ----------

from benchmark.beaver import report

md, answers = report.build_report("runs/phase0")
print(md)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 看几条改写规则的实例
# MAGIC
# MAGIC `gold_adaptations` 里记录了每条被改写的标准 SQL：用了哪条规则、改写前后的结果是否和 MySQL 一致。

# COMMAND ----------

display(spark.sql(f"""
    SELECT adaptation_rule, semantic_validation, COUNT(*) AS n
    FROM `{catalog}`.benchmark.gold_adaptations GROUP BY ALL ORDER BY n DESC
"""))

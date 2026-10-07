# Databricks notebook source
# MAGIC %md
# MAGIC # 步骤 5：自检 v2
# MAGIC
# MAGIC v1 只看执行层面，“能执行但答错”的题全部放行。这一步尝试在不看标准答案的前提下发现更多问题：
# MAGIC - **结构检查**（`loop_engineer/checks.py`）：关联条件恒为真、JOIN 没有条件、问“每个”却没分组、四舍五入与题意不符；
# MAGIC - **数值一致性检查**：把输出列追溯到产生它的统计函数，由代码核对必然成立的关系（最小值 ≤ 平均值 ≤ 最大值等），不调用 LLM。
# MAGIC
# MAGIC **原则**：每个候选信号先离线评估抓错率和误报率，只有误报接近 0 的才允许触发修复，其余只作提示。

# COMMAND ----------

# MAGIC %pip install -q -r requirements-notebook.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 检查规则的单元测试

# COMMAND ----------

assert run_tests("tests/test_checks.py") == 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 离线评估：每个信号的抓错率和误报率
# MAGIC
# MAGIC 错题取自你前面跑过的开发集运行（步骤 2 的基线、步骤 4 的内循环）；误报在开发集的标准答案和评测集以外的标准 SQL 上统计。
# MAGIC 参考：“结果有重复行”在错题上能抓到，但在正确答案上也会触发，所以不用；结构规则在标准 SQL 上误报最高 0.3%。

# COMMAND ----------

run_script("verifier_eval.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 自检 v2 下的内循环
# MAGIC
# MAGIC 和步骤 4 的结果对比：自检拦下的题、修复次数和最终结果。

# COMMAND ----------

run_script("phase6.py", "--max-repairs", "1")

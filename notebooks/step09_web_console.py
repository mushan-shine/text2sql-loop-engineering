# Databricks notebook source
# MAGIC %md
# MAGIC # 步骤 9：网页与部署（Databricks App）
# MAGIC
# MAGIC 把运行过程做成网页，每道题的每一步都能看到：
# MAGIC - **运行 Loop 页**：选题、选模型、选最大修复次数、选示例方式，点“执行”；后台走和 notebook 相同的代码路径，时间线实时刷新；
# MAGIC - **分析报告**：运行结束后由事件直接生成，不调用 LLM，附 3 张图；
# MAGIC - **Loop Debug Console**：读 Delta，按运行和题目回放每次尝试。
# MAGIC
# MAGIC 部署脚本（`scripts/deploy_app.py`）在 notebook 里运行时：沿用步骤 0 存好的 key，把代码和数据包上传到
# MAGIC `/Workspace/Users/<你>/apps/text2sql-loop-engineering`，创建 App、授予 catalog 权限并部署。
# MAGIC Free Edition 的 App 数量有限，部署前确认工作区里还能新建 App。

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-notebook.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 打包网页需要的数据
# MAGIC
# MAGIC 表结构目录、固定示例、开发集、相似题示例库、数仓使用说明写进 `app/bundle/`（含 BEAVER 内容，不提交到 git，只上传到你自己的工作区）。

# COMMAND ----------

run_script("build_app_bundle.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 部署
# MAGIC
# MAGIC 第一次部署约 5–10 分钟（要创建 App 的计算资源）。最后一行打印网页地址，点开即可使用。

# COMMAND ----------

run_script("deploy_app.py")

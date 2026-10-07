# Databricks notebook source
# MAGIC %md
# MAGIC # 步骤 0：环境准备
# MAGIC
# MAGIC 整个项目都在 Databricks 上运行，不需要在自己电脑上安装 Python 或 MySQL。这一步只做一次：
# MAGIC
# MAGIC 1. 安装 notebook 需要的几个 Python 包；
# MAGIC 2. 把大模型的 API key 存进 Databricks secret（代码里不出现 key）；
# MAGIC 3. 跑一遍测试，确认代码在这个环境里能正常工作。
# MAGIC
# MAGIC **开始前**：在左侧 **Workspace → Create → Git folder**，用仓库地址
# MAGIC `https://github.com/mushan-shine/text2sql-loop-engineering` 克隆项目，然后打开 `notebooks/step00_setup`。
# MAGIC 右上角的计算资源选 **Serverless**。

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-notebook.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 保存 API key
# MAGIC
# MAGIC 运行下面这格后，notebook 顶部会出现两个输入框：
# MAGIC - `zhipu_key`：智谱开放平台的 key（https://open.bigmodel.cn → 控制台「API Keys」），默认模型 glm-4-flash 免费；
# MAGIC - `deepseek_key`：DeepSeek 的 key（可选，按量计费）。
# MAGIC
# MAGIC 填好后**再运行一次这格**。key 会存进名为 `text2sql-loop-engineering` 的 secret scope，然后输入框自动清除。
# MAGIC 已经存过的 key，输入框留空即可，不会被覆盖。

# COMMAND ----------

from databricks.sdk import WorkspaceClient

dbutils.widgets.text("zhipu_key", "")
dbutils.widgets.text("deepseek_key", "")
values = {"zhipu-key": dbutils.widgets.get("zhipu_key").strip(), "deepseek-key": dbutils.widgets.get("deepseek_key").strip()}
if any(values.values()):
    w = WorkspaceClient()
    if SECRET_SCOPE not in {s.name for s in w.secrets.list_scopes()}:
        w.secrets.create_scope(SECRET_SCOPE)
    for key, val in values.items():
        if val:
            w.secrets.put_secret(SECRET_SCOPE, key, string_value=val)
    dbutils.widgets.removeAll()
    print("已保存到 secret scope:", SECRET_SCOPE)
print({k: ("已配置" if v else "未配置") for k, v in load_secrets().items()})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 运行测试
# MAGIC
# MAGIC 测试不连数据库、不调用模型，几秒钟跑完。最后一行应显示 `passed`，没有 `failed`。

# COMMAND ----------

assert run_tests() == 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 检查运行环境
# MAGIC
# MAGIC 在 notebook 里，代码直接使用工作区的身份认证，不需要 `databricks auth login`；SQL 由工作区里的
# MAGIC SQL warehouse 执行（Free Edition 自带 Serverless Starter Warehouse，没有查询时自动停止）。

# COMMAND ----------

from dbx.runtime import in_databricks, project_catalog
from execution.databricks_sql import DatabricksSqlExecutor

print("在 Databricks 上运行:", in_databricks())
print("项目 catalog:", project_catalog())
dbx = DatabricksSqlExecutor(catalog=project_catalog())
print("SQL warehouse:", dbx.http_path, "→", dbx.run("SELECT current_version().dbsql_version")[0][0])
dbx.close()

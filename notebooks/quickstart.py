# Databricks notebook source
# MAGIC %md
# MAGIC # 快速上手：一次跑通完整系统
# MAGIC
# MAGIC 一个 notebook 跑完主线，约 40 分钟（第一次导入数据约占 10–15 分钟）：
# MAGIC
# MAGIC 1. 保存 API key、导入课程数据；
# MAGIC 2. **基础版**：BM25 检索 + 固定示例，单次生成；
# MAGIC 3. **完整版**：相似题示例 + 数仓使用说明 + 内循环（自检 → 诊断 → 定向修复）；
# MAGIC 4. 对比两者，看一道题的修复过程，发布到 Delta / MLflow；
# MAGIC 5. （可选）部署网页。
# MAGIC
# MAGIC 用的是最终代码（`main` 分支）。评测集、干预实验、外循环和 DeepSeek 不在这里运行，见分步学习（从分支 `step-00` 开始）。
# MAGIC
# MAGIC **开始前**：
# MAGIC - 在左侧 **Workspace → Create → Git folder** 克隆 `https://github.com/mushan-shine/text2sql-loop-engineering`，
# MAGIC   名称填 `text2sql-quickstart`（和分步学习用的 Git folder 分开，两边的运行结果互不覆盖）；
# MAGIC - 打开 `notebooks/quickstart`，右上角计算资源选 **Serverless**；弹出环境版本提示时选 **Keep environment v2**；
# MAGIC - 点 **Run all**。第一次会停在“保存 API key”：在顶部 `zhipu_key` 填入智谱的 key，再点一次 Run all。

# COMMAND ----------

# MAGIC %pip install -q -r requirements-notebook.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 保存 API key
# MAGIC
# MAGIC `zhipu_key` 必填（智谱开放平台 https://open.bigmodel.cn → 控制台「API Keys」，glm-4-flash 免费）。
# MAGIC key 存进 secret scope `text2sql-loop-engineering` 后输入框自动清除；已经存过的，留空即可。
# MAGIC
# MAGIC `deploy_app` 选“是”时，最后一节会部署网页。

# COMMAND ----------

from databricks.sdk import WorkspaceClient

dbutils.widgets.text("zhipu_key", "")
dbutils.widgets.dropdown("deploy_app", "否", ["否", "是"])
key = dbutils.widgets.get("zhipu_key").strip()
if key:
    w = WorkspaceClient()
    if SECRET_SCOPE not in {s.name for s in w.secrets.list_scopes()}:
        w.secrets.create_scope(SECRET_SCOPE)
    w.secrets.put_secret(SECRET_SCOPE, "zhipu-key", string_value=key)
    dbutils.widgets.remove("zhipu_key")
    print("已保存到 secret scope:", SECRET_SCOPE)
if not load_secrets()["ZHIPUAI_API_KEY"]:
    raise RuntimeError("还没有智谱的 key：在 notebook 顶部的 zhipu_key 填入后，再点一次 Run all。")
print("智谱 key：已配置")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 导入课程数据
# MAGIC
# MAGIC 从 HuggingFace 下载课程数据包，在你的工作区建 catalog `text2sql_loop`、schema 和表（`dw` 97 张数仓表，`benchmark` 6 张评测表）。
# MAGIC 已经导入过的表自动跳过。

# COMMAND ----------

run_script("load_course_data.py")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 基础版：固定示例，单次生成
# MAGIC
# MAGIC 开发集 30 题，每题：BM25 检索候选表 → 固定示例 + 表结构拼成 prompt → glm-4-flash 生成一条 SQL → 执行 → 与标准答案比对。约 8 分钟。

# COMMAND ----------

run_script("phase1.py", "dev")
base_run = latest_run("phase1", "baseline-", mode="dev")
base = show_summary(base_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 完整版：生成端优化 + 内循环
# MAGIC
# MAGIC - 生成端：从训练集检索最相似的已解题作示例（相似题示例），并附上从训练集统计出的数仓使用说明；
# MAGIC - 内循环：第一次的 SQL 执行后自检；没通过时，在看不到标准答案的前提下诊断失败原因，路由到对应的修复技能，再生成一次。
# MAGIC
# MAGIC 先从训练集统计使用说明（只用训练集），再运行。约 15 分钟。

# COMMAND ----------

run_script("build_knowledge.py")
run_script("phase6.py", "--few-shot", "dynamic", "--knowledge", "on", "--max-repairs", "1")
loop_run = latest_run("phase6")
loop = show_summary(loop_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 对比
# MAGIC
# MAGIC 三列分开看，才能说清楚提升来自哪里：
# MAGIC - **基础版 → 完整版第一次生成**：生成端优化（相似题示例 + 使用说明）的效果；
# MAGIC - **第一次生成 → 内循环之后**：Loop 的效果。“修复成功”是第一次错、最终对的题，“误伤”是第一次对、最终错的题。

# COMMAND ----------

import pandas as pd

n = base["cases"]
base_exec = round(base["executable_rate"] * n)
display(pd.DataFrame([
    {"指标": "答对", "基础版": base["correct"], "完整版·第一次生成": loop["first_correct"], "完整版·内循环之后": loop["final_correct"]},
    {"指标": "能执行", "基础版": base_exec, "完整版·第一次生成": loop["executable_first"], "完整版·内循环之后": loop["executable_final"]},
    {"指标": "修复成功", "基础版": "—", "完整版·第一次生成": "—", "完整版·内循环之后": loop["recovered"]},
    {"指标": "误伤", "基础版": "—", "完整版·第一次生成": "—", "完整版·内循环之后": loop["harmed"]},
]).astype(str))
print(f"开发集 {n} 题。自检结果（第一次生成）：", json.dumps(loop["verifier_confusion"], ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 看一道题的修复过程
# MAGIC
# MAGIC 优先展示修复后答对的题；没有的话，展示第一道经历了修复的题。

# COMMAND ----------

records = [json.loads(line) for line in (REPO / loop_run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
repaired = [r for r in records if len(r["attempts"]) > 1]
pick = next((r for r in repaired if r["final_correct"] and not r["first_correct"]), repaired[0] if repaired else None)
if pick is None:
    print("这次运行没有题目进入修复。")
else:
    print("题目:", pick["case_id"], "| 最终", "答对" if pick["final_correct"] else "未答对")
    for a in pick["attempts"]:
        print(f"\n第 {a['attempt_id']} 次：执行={a['execution_status']}  自检={a.get('verifier_decision')}  "
              f"失败类型={a.get('failure_type')}  修复技能={a.get('repair_skill')}")
        print(a["generated_sql"][:600])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. 发布到 Delta 和 MLflow
# MAGIC
# MAGIC 两次运行写进 `traces`、`evaluation` 两个 schema（每道题的 trace、对错，以及每次运行一行汇总），并记到 MLflow（左侧 **Experiments** 可以看到）。

# COMMAND ----------

run_script("publish_run.py", base_run, "--experiment-id", "baseline")
run_script("publish_run.py", loop_run, "--experiment-id", "targeted_loop")
from dbx.runtime import project_catalog
run_ids = [json.loads((REPO / d / "run_meta.json").read_text(encoding="utf-8"))["run_id"] for d in (base_run, loop_run)]
display(spark.sql(f"""
    SELECT experiment_id, run_id, prompt_version, cases, correct, executable_rate, mlflow_run_id
    FROM `{project_catalog()}`.evaluation.runs
    WHERE run_id IN ({", ".join(f"'{r}'" for r in run_ids)})
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. 部署网页（可选）
# MAGIC
# MAGIC 顶部 `deploy_app` 选“是”才会运行。第一次部署约 5–10 分钟，最后一行打印网页地址。
# MAGIC Free Edition 的 App 数量有限，部署前在左侧 **Compute → Apps** 确认还能新建；已经有同名 App 时会更新它。
# MAGIC
# MAGIC 打开网页后，在“运行 Loop”页选几道开发集题目点“执行”，可以看到每一步的时间线和分析报告。

# COMMAND ----------

if dbutils.widgets.get("deploy_app") == "是":
    run_script("build_app_bundle.py")
    run_script("deploy_app.py")
else:
    print("已跳过部署。需要时在顶部 deploy_app 选“是”，再运行这一格。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 下一步
# MAGIC
# MAGIC 想知道每一部分为什么这样做、单独带来多少提升，回到分步学习：在分步学习用的 Git folder 里切到分支 `step-00`，
# MAGIC 按 `notebooks/step00_setup` → `step10_outer_loop` 的顺序运行。

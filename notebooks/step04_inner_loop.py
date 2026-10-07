# Databricks notebook source
# MAGIC %md
# MAGIC # 步骤 4：内循环基础版
# MAGIC
# MAGIC 第一次生成失败时，让 Agent 在**看不到标准答案**的前提下，自己发现失败、判断原因、有针对性地修复：
# MAGIC
# MAGIC ```
# MAGIC 执行 → 自检 ─通过或次数用完→ 选出最终答案
# MAGIC           └未通过→ 观察 → 诊断 → 路由 → 修复技能 → 下一次尝试
# MAGIC ```
# MAGIC
# MAGIC - 自检 v1（`loop_engineer/verifier.py`）：没有 SQL、执行报错、结果过大、空结果、整列为空；
# MAGIC - 观察（`observer.py`）：只读 12 个白名单字段，并从报错中解析错误类别、找不到的列、候选列；
# MAGIC - 诊断（`diagnose.py`）：规则优先，规则判断不了才调用 LLM；
# MAGIC - 路由（`policy.py`）和 5 个修复技能（`skills/`）：先做确定性修改，解决不了再调一次 LLM。

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-notebook.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

dev_run = latest_run("phase1", "baseline-", mode="dev")
print(dev_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 诊断准确率
# MAGIC
# MAGIC 用步骤 3 的失败标注作参照（需要先运行步骤 3 的 `label`）。参考（评测集）：规则判断的宽松准确率 76.5%，交给 LLM 的严格准确率 0/19。

# COMMAND ----------

run_script("phase4.py", dev_run, "--llm")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 修复技能单独评测
# MAGIC
# MAGIC 对基线的每道错题：观察 → 诊断 → 路由 → 修复 → 执行 → 判分。参考：能执行 4 → 8。

# COMMAND ----------

run_script("phase5.py", dev_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 完整内循环（最多修复 1 次）
# MAGIC
# MAGIC 参考：能执行 4 → 9，答对 0 → 0。

# COMMAND ----------

run_script("phase6.py", "--max-repairs", "1")

# COMMAND ----------

loop_run = latest_run("phase6")
print(loop_run)
for line in (REPO / loop_run / "results.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    if len(r["attempts"]) > 1:
        for a in r["attempts"]:
            print(f"第 {a['attempt_id']} 次：{a['execution_status']}  失败类型={a.get('failure_type')}  修复技能={a.get('repair_skill')}")
            print(a["generated_sql"][:400], "\n")
        break

# COMMAND ----------

run_script("publish_run.py", loop_run, "--experiment-id", "targeted_loop")

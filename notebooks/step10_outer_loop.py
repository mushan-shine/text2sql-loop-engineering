# Databricks notebook source
# MAGIC %md
# MAGIC # 步骤 10：外循环
# MAGIC
# MAGIC 内循环修正的是当前这道题。外循环让系统从错题中学到这个数仓的使用知识，改进之后的所有题。一轮迭代
# MAGIC （`evaluation/outer_loop.py`、`scripts/outer_loop.py`）：
# MAGIC 1. 从训练集抽题（排除评测集和开发集），用当前系统作答，对照标准答案判分并标出错因；
# MAGIC 2. 从错题中挖掘候选知识：选表偏好、补表提示、关联规则；方向相反的选表偏好按支持度处理；
# MAGIC 3. **逐条回归门禁**：在另留的题上比较“当前系统”和“当前系统 + 这一条”，答对不减少、零误伤、token 增幅 ≤ 20% 才算通过；
# MAGIC 4. 写出提案文件，每条附证据、回归结果和建议，由人决定是否采用，不会自动上线。

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-notebook.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 小规模迭代
# MAGIC
# MAGIC 60 道训练题挖掘，30 题开发集做门禁。glm 免费，但要调用几百次模型、执行几百条 SQL，约需 30–60 分钟。
# MAGIC 完整规模是 `--train-n 200`（另留 200 题验证集做门禁），需要几个小时。

# COMMAND ----------

run_script("outer_loop.py", "--train-n", "60", "--val-n", "0")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 查看提案

# COMMAND ----------

batch = latest_run("outer_loop", "batch-")
p = json.loads((REPO / batch / "proposals.json").read_text(encoding="utf-8"))
print(json.dumps(p["batch"], indent=1, ensure_ascii=False, default=str)[:2000])
for c, reg in zip(p["candidates"], p["per_item"]):
    print(f"- [{c['kind']}] {c['title']}\n  影响 {len(reg['affected'])} 题，答对 {reg['correct_before']} → {reg['correct_after']}，"
          f"新答对 {reg['fixed']}，误伤 {reg['harmed']}，通过门禁: {reg['gate_passed']}")

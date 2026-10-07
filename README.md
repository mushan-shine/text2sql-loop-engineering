# text2sql-loop-engineering

在 Databricks 上，从零开始一步步搭建一个**自愈式 Text-to-SQL Agent**：
先把企业数仓基准（BEAVER）搬到 Databricks 上并确认标准答案可信，再做最基础的“检索 + 生成”，
然后加上内循环（自检 → 诊断 → 定向修复）、生成端优化和外循环（从错题中学习）。

每一步都是一个独立的分支（`step-00` … `step-10`）和 PR，都能运行、都有测试，PR 描述里写明这一步的目的、做法和效果数据。

## 怎么运行

**全部在 Databricks 上完成，只需要浏览器**：在 Databricks 里用 Git folder 克隆本仓库，切到某一步的分支，打开对应的 notebook，Serverless 计算上点 Run all。
详细步骤和每一步的预期结果见 [执行指南](docs/手动执行与截图指南.md)。

- 需要：Databricks 账号（Free Edition 即可）、智谱 API key（glm-4-flash 免费）；DeepSeek key 可选。
- BEAVER 数据以课程数据包的形式提供（BEAVER 为 MIT 许可），步骤 1 的 notebook 自动下载并导入，不需要 MySQL。
- 也可以在本地运行脚本，每个 notebook 里的 `run_script("x.py", ...)` 就是 `python scripts/x.py ...`。

## 步骤

| 步骤 | 内容 | 板块 | 分支 / Notebook |
|---|---|---|---|
| 0 | 工程骨架：Databricks 执行器、Delta 读写、测试框架 | 平台 | `step-00` / `notebooks/step00_setup` |
| 1 | 数据底座：BEAVER 导入、复制到 Delta、双引擎比对、两条适配规则、冻结标准答案 | 数据 | `step-01` / `notebooks/step01_data` |
| 2 | 基础版：BM25 检索 + 固定示例单次生成 + 判分 | 检索、生成 | `step-02` / `notebooks/step02_baseline` |
| 3 | 可观测与评测分析：trace 发布、失败标注、干预实验 | 评测 | `step-03` / `notebooks/step03_observability` |
| 4 | 内循环基础版：自检、观察、规则诊断、路由、5 个修复技能 | 自检、诊断、修复 | `step-04` / `notebooks/step04_inner_loop` |
| 5 | 自检 v2：结构规则、数值一致性 | 自检 | `step-05` / `notebooks/step05_self_check` |
| 6 | 相似题示例 | 生成 | `step-06` / `notebooks/step06_dynamic_few_shot` |
| 7 | 数仓使用说明 | 生成 | `step-07` / `notebooks/step07_usage_notes` |
| 8 | UNION 排序规则修复、修复结果重复时提前结束 | 生成、诊断、修复 | `step-08` / `notebooks/step08_collation_early_stop` |
| 9 | 网页：运行 Loop、Debug Console、分析报告、部署为 Databricks App | 可观测 | `step-09` / `notebooks/step09_web_console` |
| 10 | 外循环：从错题挖掘知识、验证集回归门禁 | 外循环 | `step-10` / `notebooks/step10_outer_loop` |

## 步骤 0：工程骨架

**目的**：先把“在 Databricks 上执行 SQL、把结果写进 Delta”这两件基础的事做好，后面每一步都依赖它们。

**内容**

- `execution/`：引擎无关的执行接口 `ExecutionResult`；Databricks SQL Warehouse 执行器和 MySQL 执行器。
  - 只允许单条只读语句（`is_read_only`）。
  - 所有查询使用相同的会话设置：关闭 ANSI 模式（与 MySQL 的 `x/0 = NULL` 一致）、语句超时、关闭结果缓存。
  - 报错时提取错误类别（如 `UNRESOLVED_COLUMN`），后面的诊断靠它。
- `dbx/catalog.py`：Unity Catalog 布局（`benchmark` / `traces` / `evaluation` 三个 schema）和 Delta 写入：先把数据写成 parquet 上传到 UC Volume，再用一条 `INSERT ... BY NAME` 导入。
- `dbx/tables.py`：数据底座各张 Delta 表的 Arrow schema，字段定义只在这里维护。

**运行**

**Notebook**：分支 `step-00`，`notebooks/step00_setup`。

本地运行：

```bash
pip install -e ".[local,dev]"
cp .env.example .env        # 填 MySQL 和 LLM 的 key；Databricks 认证用 ~/.databrickscfg
pytest
```

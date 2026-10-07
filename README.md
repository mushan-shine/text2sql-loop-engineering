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

## 步骤 1：数据底座

**目的**：BEAVER 官方用 MySQL 执行标准 SQL，本项目跑在 Databricks 上。换了引擎，标准答案还成不成立？这一步不做，后面所有准确率都不可信。

**做法**

1. 下载 BEAVER（HuggingFace 门控数据集，需要自己申请访问），在本地 MySQL 还原，作为参照引擎（`scripts/fetch_beaver_db.py`）。
2. 把 dw 库的 97 张表复制到 Delta（`benchmark/beaver/replicate.py`）：按列映射类型；MySQL 的 `_ci`（不区分大小写）排序规则映射为 Databricks 的 `UTF8_LCASE`；逐表核对行数和列画像。
3. 每道题的标准 SQL 在两个引擎上各执行多次，比较结果（`compatibility.py`、`evaluator.py`）。
4. 结果不一致的，查明原因，只加有文档依据、并经过结果验证的改写规则（`adapter.py`），不改原始标准 SQL。
5. 通过的题把标准答案结果冻结到 `benchmark.gold_results`，之后判分只读冻结结果。

**效果**

| 阶段 | 结果 |
|---|---|
| 原样在两个引擎上严格比较 | 100 题中 37 题一致 |
| 查明三类方言差异：数值表示精度、统计函数口径（MySQL 的 `STDDEV` 是总体标准差）、排名类窗口函数的窗口帧 | — |
| 比较口径：DECIMAL 按各自精度、浮点相对误差 1e-6；补两条改写规则并逐题验证 | 可用 89 题（原样 46 + 改写 43） |

**运行**

**Notebook**：分支 `step-01`，`notebooks/step01_data`。notebook 里导入课程数据包（`scripts/load_course_data.py`：从 HuggingFace 下载这一步的产出，按原来的列类型和排序规则建表导入），不需要 MySQL，然后查看比对报告。

本地完整流程（需要 MySQL 和 BEAVER 的 HuggingFace 访问权限）：

```bash
python scripts/fetch_beaver_db.py      # 下载并在 MySQL 还原 BEAVER（需要 HuggingFace 访问权限）
python scripts/phase0.py               # 复制、比对、冻结标准答案
```

`data/`、`runs/` 等含 BEAVER 内容的目录不入库（见 `.gitignore`）。

## 步骤 2：基础版（检索 + 单次生成）

**目的**：做出最简单、能端到端运行的 Text-to-SQL，作为后面所有优化的对照基线。

**做法**

- **检索**（`agent/retriever.py`）：BM25 从 97 张表中选出 20 张候选表。每张表的“文档”由表名、列名、样例值组成，权重 3 / 2 / 1；分数相同时按表名排序，结果确定。
- **生成**（`agent/generator.py`）：prompt = 规则 + 候选表的 schema + 固定 3 个示例 + 问题，生成一次，不修复。
  - 规则写的是这个数仓的通用约定（方言、字符串不区分大小写、MySQL 统计函数在 Databricks 上的对应写法），不含任何具体题目的信息。
  - 固定示例取自评测集以外的题。
- **LLM 客户端**（`agent/llm.py`）：OpenAI 兼容接口，可切换智谱 glm-4-flash 和 DeepSeek；贪心解码 + 按 prompt 指纹缓存，同一配置重跑结果相同；调用次数和 token 有上限。
- **数据集**：评测集 89 题只在配置冻结后运行；另从评测集以外抽 30 题作为开发集（`evaluation/devset.py`，标准答案在 MySQL 上实时计算），所有调参只在开发集上做。
- **判分**（`evaluation/baseline.py`）：生成结束后才用标准答案判分。

**效果**

| 实验 | 条件 | 结果 |
|---|---|---|
| 候选表数 k | 评测集以外 300 题 | k=10 / 15 / 20 / 25 / 30 时表召回 0.73 / 0.84 / **0.91** / 0.94 / 0.95，取拐点 k=20 |
| 开发集基线 | glm-4-flash，30 题 | 答对 0，能执行 4；表召回 0.92 |
| 开发集基线 | DeepSeek，30 题 | 答对 3，能执行 24 |
| 评测集基线（配置冻结后跑一次） | glm-4-flash，89 题 | 答对 2，能执行 21 |

glm 的 26 个报错中，21 个是“列找不到”，其中 20 个这一列其实在另一张已选中的表里，只是挂错了表别名。这类错误有明确的报错信号，是下一阶段内循环的修复对象。

**运行**

**Notebook**：分支 `step-02`，`notebooks/step02_baseline`。

本地运行：

```bash
python scripts/phase1.py build-dev     # 构建开发集
python scripts/phase1.py dev           # 开发集上跑基线
python scripts/phase1.py baseline      # 评测集（配置冻结后）
```

## 步骤 3：可观测与评测分析

**目的**：基线跑出来以后，要能回答两个问题：每道题具体发生了什么（trace），以及错在哪里、为什么错（失败分析）。后面每一项优化都从这里找方向。

**做法**

- **trace 与发布**（`dbx/publish.py`、`scripts/publish_run.py`）：
  - 每次尝试写一行 `traces.execution_traces`，只放 Agent 自己看得到的内容；
  - 用标准答案算出的对错、表召回率写进 `evaluation.*`，和 trace 物理分开。后面内循环会读 trace，这样它从存储层面就看不到答案；
  - 汇总写进 `evaluation.runs` 和 MLflow；同一个 run_id 重复发布时先删旧数据，保证幂等。
- **失败标注器**（`benchmark/beaver/subtasks.py`，只用于离线评测）：把生成的 SQL 和标准 SQL 都解析成语法树（`agent/sql_analysis.py`），逐项比较表、列、关联、常量、运算，给错题标出主因和多标签。
- **干预实验**（`evaluation/intervention.py`）：每次只把标准答案中的一类信息（表、列、关联键、领域知识、查询拆解）提示给模型，看能不能修好。用来区分“缺信息”还是“模型能力不够”，结果只作上界参考。

**效果**

| 分析 | 条件 | 结果 |
|---|---|---|
| 标注器自检 | 标准 SQL 对照自身 | 开发集 30/30、评测样本 100/100 无误报；人工抽检 20 题一致率严格 75%、宽松 90% |
| 失败主因 | glm，评测集 87 个失败 | 选表 76、列映射 7、执行 2、关联键 1、领域知识 1 |
| 缺失的标准答案用表 | 同上 | 已检索到但没用上 132 次，没检索到 32 次 |
| 干预：5 类信息全部提示 | glm，开发集错题 | 修好 0/30 |
| 干预：5 类信息全部提示 | DeepSeek，开发集 27 道错题 | 修好 7/27 |

两条结论决定了后面的方向：
1. 缺失的表大多已经检索到了，问题主要在生成端“选错表、挂错列”，不在检索。
2. glm 即使拿到标准答案的全部信息也修不好，瓶颈是模型能力；DeepSeek 拿到信息能修好一部分，说明对它来说信息不足才是主要问题，Loop 有发挥空间。

**运行**

**Notebook**：分支 `step-03`，`notebooks/step03_observability`。

本地运行：

```bash
python scripts/publish_run.py runs/phase1/<run_id>          # 发布一次运行
python scripts/phase3.py label runs/phase1/<run_id>          # 失败标注
python scripts/phase3.py intervene runs/phase1/<dev_run_id>  # 干预实验（只在开发集）
```

## 步骤 4：内循环基础版

**目的**：第一次生成失败时，让 Agent 在**看不到标准答案**的前提下，自己发现失败、判断原因、有针对性地修复。

**流程**（`loop_engineer/controller.py`）

```
执行 → 自检 ─通过或次数用完→ 选出最终答案
          └未通过→ 观察 → 诊断 → 路由 → 修复技能 → 下一次尝试
```

**做法**

- **自检 v1**（`verifier.py`）：只看执行层面的 5 个信号：没有 SQL、执行报错、结果过大、空结果、整列为空。“通过”只表示没发现问题，不代表答案正确。
- **观察**（`observer.py`）：只按白名单读取尝试记录里的 12 个字段，并从报错文本中解析出错误类别、找不到的列和别名、数据库给出的候选列、不存在的表。用白名单而不是黑名单，标准答案相关的字段天然进不来。
- **诊断**（`diagnose.py`）：规则优先，规则判断不了才调用 LLM。核心规则是：列找不到时，在 schema 里查这一列属于哪些表——在 SQL 已用的表里，是别名挂错；在检索到但没用的表里，是漏用了表。
- **路由**（`policy.py`）：失败类型 → 技能的映射写成字典，禁用某个技能时退回 RepairSQL。
- **5 个修复技能**（`skills/`）：先做不需要 LLM 的确定性修改，解决不了的部分再带着定向指令调用一次 LLM。
  - SchemaSearch（列映射）：按作用域把挂错别名的列改到唯一拥有它的表；如果改完后关联条件变成 `x = x`，撤回这处修改。
  - RetrieveAgain（选表）、FindJoinPath（关联键）、ReplanQuery（查询拆解）、RepairSQL（执行错误、兜底）。
- **最终答案**：最后一个自检通过的 > 最后一个能执行的 > 最后一次。
- **防泄露**：测试强制诊断、路由、观察和修复模块不能导入基准与评测代码；判分在 Loop 返回之后才做（`evaluation/loop_run.py`）。

**效果**

| 实验 | 条件 | 结果 |
|---|---|---|
| 诊断准确率 | 评测集 87 个失败，参照失败标注器 | 整体严格 40.2% / 宽松 66.7%；规则判断的 68 个宽松 76.5%；规则判断不了、交给 LLM 的 19 个严格 0/19 |
| 规则覆盖率 | glm，开发集 | 27 次诊断中 26 次由规则完成 |
| 修复技能单独评测 | glm，开发集 30 个失败 | 能执行 4 → 8 |
| 完整内循环 | glm，开发集，最多修复 1 次 | 能执行 4 → 9，答对 0 → 0；自检拦下 27、漏报 3 |

**踩过的坑**：SchemaSearch 一开始会把关联条件改成 `sd.X = sd.X`，SQL 能执行但两张表失去关联，答案是错的，统计出的“能执行”因此虚高（4 → 10，修正后 4 → 8）。加了“改完变成 `x = x` 就撤回”的守卫，并补了回归测试。

修复让更多 SQL 能跑通，但答对的题没有增加：glm 的干预实验已经表明它受限于模型能力。另外，能执行但答错的题，v1 的自检全部放行，进不了修复环节。下一步加强自检。

**运行**

**Notebook**：分支 `step-04`，`notebooks/step04_inner_loop`。

本地运行：

```bash
python scripts/phase4.py runs/phase1/<run_id> --llm   # 诊断准确率（需先跑步骤 3 的 label）
python scripts/phase5.py runs/phase1/<dev_run_id>       # 修复技能单独评测
python scripts/phase6.py --max-repairs 1                      # 内循环，开发集
python scripts/phase6.py --verifier oracle --max-repairs 1    # 上界：用标准答案当自检器
```

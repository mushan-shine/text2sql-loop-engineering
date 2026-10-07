"""Console analysis report + chart data: built from real loop events (stub LLM / executor)."""
from agent.generator import FewShotGenerator
from agent.retriever import BM25TableRetriever
from app.report import attempt_progress, build_report, cases_from_events, first_final, outcome, status_grid
from benchmark.beaver.dataset import BeaverCase
from evaluation.loop_run import run_arm
from execution.base import ExecutionResult
from loop_engineer.controller import LoopConfig, LoopController
from loop_engineer.diagnose import Diagnoser
from loop_engineer.policy import Policy
from loop_engineer.verifier import SelfVerifier
from skills.base import RepairContext
from tests.test_phase6 import BAD, CAT, ERR, Chat, controller

CASE = BeaverCase(case_id="dw:1", split="dw", question="average enrollment per department", db="dw", gold_sql="")
REQ = {"model": "glm-4-flash", "strategy": "targeted", "verifier": "self", "split": "dev", "max_repairs": 1}


class AlwaysError:
    def execute(self, sql, db, max_rows=None):
        return ExecutionResult("databricks", "ERROR", error=ERR, error_class="UNRESOLVED_COLUMN")


def _run(tmp_path, ctl, judge_ok: bool):
    events = []
    run_id, summary = run_arm([CASE], {"dw:1": lambda rows: (judge_ok, "")}, ctl, lambda cid: SelfVerifier(),
                              "console-targeted-self", tmp_path, {}, on_event=lambda c, s, p: events.append(
                                  {"t": float(len(events)), "case_id": c, "step": s, **p}))
    return events, summary, run_id


def test_outcome_uses_the_chosen_final_attempt():
    assert outcome(False, True, "ERROR", "SUCCESS") == "已恢复"
    assert outcome(True, False, "SUCCESS", "ERROR") == "被误伤"
    assert outcome(False, False, "ERROR", "SUCCESS") == "报错 → 可执行（仍错）"
    assert outcome(False, False, "ERROR", "ERROR") == "仍报错"


def test_report_for_a_deterministic_repair_that_runs_but_is_wrong(tmp_path):
    events, summary, run_id = _run(tmp_path, controller(Chat(lambda n: f"```sql\n{BAD}\n```")), judge_ok=False)
    cases = cases_from_events(events)
    assert len(cases) == 1 and cases[0]["outcome"] == "报错 → 可执行（仍错）"
    md = build_report(events, REQ, summary, run_id, 12.0)
    assert "# Loop 运行分析报告" in md and "| 可执行 | 0 / 1 | 1 / 1 |" in md
    assert "d.NUM_ENROLLED -> s.NUM_ENROLLED" in md          # skill internals reach the report
    assert "确定性修复有效" in md and "Verifier 是当前短板" in md  # rule-based findings
    # self-check passed but gold says wrong -> reported as a verifier miss, never as "PASS" = correct
    assert "| 自检通过但答案错（Verifier 漏报） | 1 |" in md and "第 2 次 · 自检通过 | 是 |" in md
    assert "PASS" not in md.split("## 7.")[0]
    assert "```diff" in md


def test_report_counts_a_recovery(tmp_path):
    events, summary, run_id = _run(tmp_path, controller(Chat(lambda n: f"```sql\n{BAD}\n```")), judge_ok=True)
    md = build_report(events, {**REQ, "model": "deepseek-flash"}, summary, run_id)
    # attempt 1 errored (never correct), the repaired attempt 2 is judged right -> one recovery
    assert "| 恢复 / 误伤 / 净收益 | 1 / 0 / +1 |" in md and "已恢复" in md and "Loop 修好了 1 题" in md
    assert "Verifier 是当前短板" not in md


def test_multiple_repair_rounds_and_chart_data(tmp_path):
    ctl = LoopController(BM25TableRetriever(CAT), FewShotGenerator(Chat(lambda n: ""), CAT, []), AlwaysError(),
                         Diagnoser(CAT), Policy(), RepairContext(CAT, Chat(lambda n: f"```sql\n{BAD} LIMIT {n}\n```")),
                         LoopConfig(max_attempts=3, top_k=2))  # each LLM repair differs: no early stop on a repeat
    ctl.generator = FewShotGenerator(Chat(lambda n: f"```sql\n{BAD}\n```"), CAT, [])
    events, summary, run_id = _run(tmp_path, ctl, judge_ok=False)
    c = cases_from_events(events)[0]
    assert len(c["executions"]) == 3 and len(c["repairs"]) == 2 and len(c["diagnoses"]) == 2
    assert [r["attempt_id"] for r in c["repairs"]] == [2, 3]
    prog = attempt_progress([c], 3)
    assert [r["题数"] for r in prog if r["指标"] == "可执行"] == [0, 0, 0]
    assert {(r["指标"], r["阶段"]): r["题数"] for r in first_final([c])}[("可执行", "最终")] == 0
    grid = status_grid([c])
    assert [g["状态"] for g in grid] == ["报错"] * 3 and grid[1]["由谁生成"] == "SchemaSearch"
    assert grid[-1]["最终答案"] == "是"
    md = build_report(events, {**REQ, "max_repairs": 2}, summary, run_id)
    assert "最多修复 2 次（最多 3 次 SQL 尝试）" in md and "第 2 轮修复" in md and "| 第 1 次 | 第 2 次 | 第 3 次 |" in md
    assert "修复后出现同样的报错" in md and "多轮修复的效果" in md
